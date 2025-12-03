import json
import logging
import shutil
import struct
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import kagglehub
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from ..attack import (
    ComposeImageLoss,
    GmiDiscriminatorLoss,
    ImageAugmentClassificationLoss,
    ImageClassifierAttackConfig,
    ImageClassifierAttacker,
)
from ..attack.optimize import SimpleWhiteBoxOptimization, SimpleWhiteBoxOptimizationConfig
from ..metrics import ImageFidPRDCMetric
from ..models.classifiers.base import construct_classifiers_by_name
from ..models.gans import auto_discriminator_from_pretrained, auto_generator_from_pretrained
from ..sampler import SimpleLatentsSampler

LOGGER = logging.getLogger(__name__)


DEFAULT_GAN_GENERATOR_URL = "https://drive.google.com/file/d/1is0tNjxL4QUAqjrhj7x0AjAh4fLIt-D0/view?usp=drive_link"
DEFAULT_GAN_DISCRIMINATOR_URL = "https://drive.google.com/file/d/1f9IzPbp8qucx55xN879b-kbDpZtzPpxC/view?usp=drive_link"


DATASET_CONFIGS = {
    "mnist": {
        "factory": torchvision.datasets.MNIST,
        "kwargs": {"download": True},
        "in_channels": 1,
    },
    "fmnist": {
        "factory": torchvision.datasets.FashionMNIST,
        "kwargs": {"download": True},
        "in_channels": 1,
    },
    "overhead_mnist": {
        "factory": None,
        "kwargs": {},
        "in_channels": 1,
    },
}


def _download_if_missing(url: str, path: Path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading %s to %s via gdown", path.name, path)
    downloaded_path = gdown.download(url=url, output=str(path), quiet=False, fuzzy=True)
    if downloaded_path is None or not Path(downloaded_path).exists():
        raise RuntimeError(f"Failed to download {url} to {path}")


def ensure_gan_checkpoints(
    checkpoint_dir: Path,
    generator_url: str = DEFAULT_GAN_GENERATOR_URL,
    discriminator_url: str = DEFAULT_GAN_DISCRIMINATOR_URL,
) -> tuple[Path, Path]:
    generator_path = checkpoint_dir / "G.pth"
    discriminator_path = checkpoint_dir / "D.pth"
    _download_if_missing(generator_url, generator_path)
    _download_if_missing(discriminator_url, discriminator_path)
    return generator_path, discriminator_path


def copy_gan_checkpoints_for_attack(
    model_dir: Path, generator_ckpt: Path, discriminator_ckpt: Path
) -> tuple[Path, Path]:
    target_dir = model_dir / "gan"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_generator = target_dir / generator_ckpt.name
    target_discriminator = target_dir / discriminator_ckpt.name
    if not target_generator.exists():
        shutil.copyfile(generator_ckpt, target_generator)
    if not target_discriminator.exists():
        shutil.copyfile(discriminator_ckpt, target_discriminator)
    return target_generator, target_discriminator


class PTImageDataset(torch.utils.data.Dataset):
    """Dataset helper for loading PT tensors saved by :func:`ensure_datasets_pt`.

    The class mirrors the behavior of the ad-hoc dataset used in the prototype
    script: images are converted to three channels, resized to a configurable
    resolution through PIL, and an optional transform is applied afterwards.

    Args:
        pt_path (Path): Path to a ``.pt`` file containing ``{"images": ..., "labels": ...}``.
        dataset_name (str): Name of the dataset (mnist / fmnist / cifar10) used for display.
        preprocess_resolution (int): Target resolution before applying ``transform``.
        transform (Callable, optional): Additional transform applied after resizing.
    """

    def __init__(
        self,
        pt_path: Path,
        dataset_name: str,
        preprocess_resolution: int = 64,
        transform=None,
    ) -> None:
        super().__init__()

        obj = torch.load(pt_path)
        self.images = obj["images"]
        self.labels = obj["labels"].long()

        # Normalize shapes [N, C, H, W] / [N, H, W] -> [N, C, H, W]
        if self.images.ndim == 3:
            self.images = self.images.unsqueeze(1)

        if self.images.dtype != torch.uint8:
            # preserve information even when tensors are float in [0,1]
            if self.images.max() <= 1:
                self.images = (self.images * 255).round().to(torch.uint8)
            else:
                self.images = self.images.to(torch.uint8)

        self.preprocess_resolution = preprocess_resolution
        self.transform = transform

        dataset_name = dataset_name.lower()
        if dataset_name == "mnist":
            self.name = "MNIST"
        elif dataset_name in ("fmnist", "fashionmnist", "fashion-mnist"):
            self.name = "FashionMNIST"
        elif dataset_name == "cifar10":
            self.name = "CIFAR10"
        else:
            raise ValueError(f"Unknown dataset_name: {dataset_name}")

        self.targets = self.labels.tolist()

        self._to_pil = transforms.ToPILImage()
        self._resize = transforms.Resize(self.preprocess_resolution)

    def __len__(self):
        return self.images.shape[0]

    def _preprocess_one(self, img_tensor: torch.Tensor):
        if img_tensor.ndim != 3:
            raise RuntimeError(f"Expected image tensor [C,H,W], got shape {img_tensor.shape}")

        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)
        elif img_tensor.shape[0] != 3:
            raise RuntimeError(f"Unsupported channel count: {img_tensor.shape[0]}")

        pil_img = self._to_pil(img_tensor)
        pil_img = self._resize(pil_img)
        return pil_img

    def __getitem__(self, idx: int):
        img_tensor = self.images[idx]
        label = self.labels[idx].item()

        img = self._preprocess_one(img_tensor)
        if self.transform is not None:
            img = self.transform(img)

        return img, label


@dataclass
class DatasetDescriptor:
    name: str
    in_channels: int
    train_images: torch.Tensor
    train_labels: torch.Tensor
    test_images: torch.Tensor
    test_labels: torch.Tensor
    root_dir: Path


# --------------------- Dataset helpers ---------------------


def _copy_if_missing(src: Path, dst: Path):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _download_overhead_mnist(root: Path):
    LOGGER.info("Downloading Overhead MNIST from Kaggle via kagglehub.")
    cache_dir = Path(kagglehub.dataset_download("datamunge/overheadmnist"))
    for filename in [
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "test-images-idx3-ubyte",
        "test-labels-idx1-ubyte",
    ]:
        _copy_if_missing(cache_dir / filename, root / filename)


def _read_idx_images(path: Path) -> torch.Tensor:
    with path.open("rb") as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = torch.tensor(bytearray(f.read()), dtype=torch.uint8)
    images = data.view(num, 1, rows, cols).float() / 255.0
    return images


def _read_idx_labels(path: Path) -> torch.Tensor:
    with path.open("rb") as f:
        _, num = struct.unpack(">II", f.read(8))
        data = torch.tensor(bytearray(f.read()), dtype=torch.long)
    return data.view(num)


def _load_overhead_mnist(root: Path):
    _download_overhead_mnist(root)
    train_imgs = _read_idx_images(root / "train-images-idx3-ubyte")
    train_labels = _read_idx_labels(root / "train-labels-idx1-ubyte")
    test_imgs = _read_idx_images(root / "test-images-idx3-ubyte")
    test_labels = _read_idx_labels(root / "test-labels-idx1-ubyte")
    return train_imgs, train_labels, test_imgs, test_labels, DATASET_CONFIGS["overhead_mnist"]["in_channels"]


def _load_raw_dataset(name: str, root: Path):
    name = name.lower()
    if name not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset {name}")

    if name == "overhead_mnist":
        return _load_overhead_mnist(root)

    cfg = DATASET_CONFIGS[name]
    factory = cfg["factory"]
    kwargs = cfg["kwargs"].copy()
    kwargs["root"] = str(root)

    transform = torchvision.transforms.ToTensor()
    train_set = factory(train=True, transform=transform, **kwargs)
    test_set = factory(train=False, transform=transform, **kwargs)

    def _stack(ds):
        imgs = torch.stack([img for img, _ in ds], dim=0)
        labels = torch.tensor([lbl for _, lbl in ds], dtype=torch.long)
        return imgs, labels

    train_imgs, train_labels = _stack(train_set)
    test_imgs, test_labels = _stack(test_set)
    return train_imgs, train_labels, test_imgs, test_labels, cfg["in_channels"]


def _save_pt(path: Path, images: torch.Tensor, labels: torch.Tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"images": images, "labels": labels}, path)


def _split_public_private(labels: torch.Tensor, seed: int = 42):
    generator = torch.Generator().manual_seed(seed)
    indices = torch.arange(len(labels))
    public_idx = []
    private_idx = []
    for cls in labels.unique().tolist():
        cls_indices = indices[labels == cls]
        perm = torch.randperm(len(cls_indices), generator=generator)
        cutoff = len(cls_indices) // 2
        public_idx.append(cls_indices[perm[:cutoff]])
        private_idx.append(cls_indices[perm[cutoff:]])
    public_idx = torch.cat(public_idx)
    private_idx = torch.cat(private_idx)
    return public_idx, private_idx


def ensure_datasets_pt(
    exp_root: Path,
    split_public_private: bool = True,
    dataset_names: Optional[List[str]] = None,
):
    exp_root = Path(exp_root)
    target_names = dataset_names or ["mnist", "fmnist", "overhead_mnist"]
    for name in target_names:
        dataset_dir = exp_root / name / "dataset"
        train_path = dataset_dir / "train.pt"
        test_path = dataset_dir / "test.pt"
        if train_path.exists() and test_path.exists():
            LOGGER.info("Dataset %s already prepared.", name)
        else:
            LOGGER.info("Preparing dataset %s", name)
            train_imgs, train_labels, test_imgs, test_labels, in_channels = _load_raw_dataset(
                name, dataset_dir
            )
            _save_pt(train_path, train_imgs, train_labels)
            _save_pt(test_path, test_imgs, test_labels)
            (dataset_dir / "meta.json").write_text(
                json.dumps({"in_channels": in_channels, "name": name})
            )

        if not split_public_private:
            continue

        public_path = dataset_dir / "train_public.pt"
        private_path = dataset_dir / "train_private.pt"
        test_public_path = dataset_dir / "test_public.pt"
        test_private_path = dataset_dir / "test_private.pt"
        if all(p.exists() for p in [public_path, private_path, test_public_path, test_private_path]):
            continue

        data = torch.load(train_path)
        train_imgs, train_labels = data["images"], data["labels"]
        pub_idx, pri_idx = _split_public_private(train_labels)
        _save_pt(public_path, train_imgs[pub_idx], train_labels[pub_idx])
        _save_pt(private_path, train_imgs[pri_idx], train_labels[pri_idx])

        data = torch.load(test_path)
        test_imgs, test_labels = data["images"], data["labels"]
        pub_idx, pri_idx = _split_public_private(test_labels)
        _save_pt(test_public_path, test_imgs[pub_idx], test_labels[pub_idx])
        _save_pt(test_private_path, test_imgs[pri_idx], test_labels[pri_idx])


def load_datasets_from_pt(
    exp_root: Path, use_public: bool = True, dataset_names: Optional[List[str]] = None
) -> List[DatasetDescriptor]:
    exp_root = Path(exp_root)
    datasets: List[DatasetDescriptor] = []
    target_names = dataset_names or ["mnist", "fmnist", "overhead_mnist"]
    for name in target_names:
        dataset_dir = exp_root / name / "dataset"
        meta_path = dataset_dir / "meta.json"
        if meta_path.exists():
            in_channels = json.loads(meta_path.read_text()).get("in_channels", 1)
        else:
            # try infer from tensor shape
            data = torch.load(dataset_dir / "train.pt")
            in_channels = data["images"].shape[1]

        if use_public and (dataset_dir / "train_public.pt").exists():
            train_data = torch.load(dataset_dir / "train_public.pt")
            test_data = torch.load(dataset_dir / "test_public.pt")
        else:
            train_data = torch.load(dataset_dir / "train.pt")
            test_data = torch.load(dataset_dir / "test.pt")

        descriptor = DatasetDescriptor(
            name=name,
            in_channels=in_channels,
            train_images=train_data["images"],
            train_labels=train_data["labels"],
            test_images=test_data["images"],
            test_labels=test_data["labels"],
            root_dir=exp_root / name,
        )
        datasets.append(descriptor)
    return datasets


# --------------------- Training helpers ---------------------

def add_gaussian_noise(images: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return images
    noise = torch.randn_like(images) * sigma
    return torch.clamp(images + noise, 0.0, 1.0)


def preprocess_for_resnet(imgs: torch.Tensor, is_gray: bool) -> torch.Tensor:
    imgs = F.interpolate(imgs, size=(32, 32), mode="bilinear", align_corners=False)
    if is_gray:
        imgs = imgs.repeat(1, 3, 1, 1)
    return imgs


def _load_target_model_from_checkpoint(model_path: Path, num_classes: int) -> torch.nn.Module:
    checkpoint = torch.load(model_path, map_location="cpu")
    backbone = checkpoint.get("backbone", "resnet34")
    model = construct_classifiers_by_name(backbone, num_classes=num_classes, resolution=32)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def format_sigma(sigma: float) -> str:
    if sigma.is_integer():
        return str(int(sigma))
    return f"{sigma:.1f}".rstrip("0").rstrip(".")


@dataclass
class TrainingResult:
    model_path: Path
    metrics_path: Path
    history: Dict[str, List[float]]
    test_acc: float


def train_resnet_for_dataset(
    dataset: DatasetDescriptor,
    sigma: float,
    batch_size: int,
    epochs: int,
    num_workers: int,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> TrainingResult:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Training %s with sigma=%.2f on %s", dataset.name, sigma, device)

    train_ds = TensorDataset(dataset.train_images, dataset.train_labels)
    test_ds = TensorDataset(dataset.test_images, dataset.test_labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = construct_classifiers_by_name(
        "resnet34", num_classes=int(dataset.train_labels.max().item()) + 1, resolution=32
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    history = {"loss": [], "acc": []}
    for _ in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in tqdm(train_loader, leave=False):
            imgs = imgs.float().to(device)
            labels = labels.to(device)
            imgs = preprocess_for_resnet(imgs, dataset.in_channels == 1)
            imgs = add_gaussian_noise(imgs, sigma)

            optimizer.zero_grad()
            logits, _ = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(imgs)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(imgs)
        history["loss"].append(running_loss / total)
        history["acc"].append(correct / total)

    # evaluation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        correct = 0
        total = 0
        for imgs, labels in tqdm(test_loader, leave=False):
            imgs = imgs.float().to(device)
            labels = labels.to(device)
            imgs = preprocess_for_resnet(imgs, dataset.in_channels == 1)
            logits, _ = model(imgs)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(imgs)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    test_acc = correct / total
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    sigma_str = format_sigma(sigma)
    out_dir = dataset.root_dir / f"noise_{sigma_str}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # confusion matrix plot
    cm = confusion_matrix(all_labels.numpy(), all_preds.numpy())
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues")
    plt.title(f"{dataset.name} sigma={sigma}")
    plt.savefig(out_dir / "conf_matrix.png", bbox_inches="tight")
    plt.close()

    # train curves
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history["loss"], label="train loss")
    ax[0].set_xlabel("epoch")
    ax[0].legend()
    ax[1].plot(history["acc"], label="train acc")
    ax[1].set_xlabel("epoch")
    ax[1].legend()
    plt.suptitle(f"{dataset.name} sigma={sigma}")
    plt.savefig(out_dir / "train_curves.png", bbox_inches="tight")
    plt.close()

    # save model and metrics
    model_path = out_dir / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "backbone": "resnet34",
            "input_shape": "3x32x32",
            "sigma": sigma,
            "dataset": dataset.name,
            "num_classes": int(dataset.train_labels.max().item()) + 1,
        },
        model_path,
    )
    metrics = {
        "dataset": dataset.name,
        "sigma": sigma,
        "min_train_loss": float(min(history["loss"])),
        "train_acc_last_epoch": float(history["acc"][-1]),
        "train_acc_best": float(max(history["acc"])),
        "test_acc": float(test_acc),
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    return TrainingResult(model_path=model_path, metrics_path=metrics_path, history=history, test_acc=test_acc)


# --------------------- Evaluation helpers ---------------------

def evaluate_checkpoint(model_path: Path, dataset: DatasetDescriptor, batch_size: int, num_workers: int):
    checkpoint = torch.load(model_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = construct_classifiers_by_name(
        checkpoint.get("backbone", "resnet34"),
        num_classes=checkpoint.get("num_classes", int(dataset.train_labels.max().item()) + 1),
        resolution=32,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    test_loader = DataLoader(
        TensorDataset(dataset.test_images, dataset.test_labels),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    preds = []
    labels = []
    with torch.no_grad():
        for imgs, lbls in tqdm(test_loader, leave=False):
            imgs = preprocess_for_resnet(imgs.float().to(device), dataset.in_channels == 1)
            logits, _ = model(imgs)
            preds.append(logits.argmax(dim=1).cpu())
            labels.append(lbls)
    preds = torch.cat(preds)
    labels = torch.cat(labels)
    acc = (preds == labels).float().mean().item()
    cm = confusion_matrix(labels.numpy(), preds.numpy())
    return acc, cm


# --------------------- GMI helpers ---------------------

def run_simple_gmi(
    dataset: DatasetDescriptor,
    model_path: Path,
    sigma: float,
    batch_size: int,
    num_workers: int,
    samples_per_class: int = 10,
    mse_weight: float = 1.0,
    ce_weight: float = 1.0,
    iter_times: int = 200,
    generator_ckpt_path: Path | None = None,
    discriminator_ckpt_path: Path | None = None,
    preprocess_resolution: int = 64,
    optimize_num: int = 50,
    z_dim: int = 100,
    inner_iter_times: int = 1500,
    class_loss_weight: float = 100.0,
    disc_loss_weight: float = 1.0,
):
    if generator_ckpt_path is not None and discriminator_ckpt_path is not None:
        return _run_gmi_with_pretrained_gan(
            dataset,
            model_path,
            sigma,
            batch_size,
            num_workers,
            preprocess_resolution=preprocess_resolution,
            optimize_num=optimize_num,
            z_dim=z_dim,
            inner_iter_times=inner_iter_times,
            class_loss_weight=class_loss_weight,
            disc_loss_weight=disc_loss_weight,
            generator_ckpt_path=generator_ckpt_path,
            discriminator_ckpt_path=discriminator_ckpt_path,
        )

    checkpoint = torch.load(model_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = construct_classifiers_by_name(
        checkpoint.get("backbone", "resnet34"),
        num_classes=checkpoint.get("num_classes", int(dataset.train_labels.max().item()) + 1),
        resolution=32,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # select subset for inversion
    subset_images = []
    subset_labels = []
    for cls in dataset.test_labels.unique():
        cls_indices = (dataset.test_labels == cls).nonzero(as_tuple=True)[0][:samples_per_class]
        subset_images.append(dataset.test_images[cls_indices])
        subset_labels.append(dataset.test_labels[cls_indices])
    subset_images = torch.cat(subset_images, dim=0)
    subset_labels = torch.cat(subset_labels, dim=0)

    loader = DataLoader(TensorDataset(subset_images, subset_labels), batch_size=batch_size, shuffle=False, num_workers=num_workers)

    sampler = SimpleLatentsSampler(100, batch_size=batch_size)
    generator = construct_generator().to(device)
    optimization_config = SimpleWhiteBoxOptimizationConfig(
        experiment_dir=str(model_path.parent / "gmi"),
        device=device,
        optimizer="Adam",
        optimizer_kwargs={"lr": 0.05},
        iter_times=iter_times,
        show_loss_info_iters=max(1, iter_times // 5),
    )

    reconstructions = []
    gt_images = []
    gt_labels = []

    current_targets: Optional[torch.Tensor] = None

    def loss_fn(fake_images: torch.Tensor, labels: torch.Tensor):
        assert current_targets is not None
        resized = preprocess_for_resnet(fake_images, False)
        logits, _ = model(resized)
        ce_loss = F.cross_entropy(logits, labels) * ce_weight
        mse_loss = F.mse_loss(fake_images, current_targets) * mse_weight
        total = ce_loss + mse_loss
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean().item()
        return total, OrderedDict([
            ("ce", ce_loss.item()),
            ("mse", mse_loss.item()),
            ("acc", acc),
        ])

    optimization = SimpleWhiteBoxOptimization(optimization_config, generator, loss_fn)

    for imgs, labels in tqdm(loader, leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)
        # prepare targets at generator resolution
        targets = F.interpolate(
            imgs, size=(64, 64), mode="bilinear", align_corners=False
        )
        if dataset.in_channels == 1:
            targets = targets.repeat(1, 3, 1, 1)
        current_targets = targets

        latents_sample = sampler(list(labels.cpu().numpy()), len(labels))
        if isinstance(latents_sample, dict):
            latents_sample = next(iter(latents_sample.values()))
        latents = latents_sample.to(device)
        output = optimization(latents, labels)
        reconstructions.append(output.images.cpu())
        gt_images.append(targets.cpu())
        gt_labels.append(labels.cpu())

    reconstructions = torch.cat(reconstructions, dim=0)
    gt_images = torch.cat(gt_images, dim=0)
    gt_labels = torch.cat(gt_labels, dim=0)

    # metrics
    mse = F.mse_loss(reconstructions, gt_images, reduction="none").view(len(reconstructions), -1).mean(dim=1)
    mse_mean = mse.mean().item()
    psnr = 10 * torch.log10(1.0 / torch.clamp(mse, min=1e-8))

    try:
        from piq import ssim
    except Exception:
        # simple fallback
        def ssim(x, y):
            mu_x = x.mean(dim=(-2, -1))
            mu_y = y.mean(dim=(-2, -1))
            sigma_x = x.var(dim=(-2, -1))
            sigma_y = y.var(dim=(-2, -1))
            sigma_xy = ((x - mu_x[..., None, None]) * (y - mu_y[..., None, None])).var(dim=(-2, -1))
            c1 = 0.01 ** 2
            c2 = 0.03 ** 2
            ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))
            return ssim_map.mean(dim=1)
    ssim_scores = ssim(reconstructions, gt_images)
    if isinstance(ssim_scores, torch.Tensor):
        ssim_scores = ssim_scores.cpu()
    ssim_mean = float(torch.tensor(ssim_scores).mean().item())

    out_dir = model_path.parent / "gmi"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_image(make_grid(reconstructions, nrow=batch_size, normalize=True), out_dir / "reconstructions.png")

    metrics = {
        "dataset": dataset.name,
        "sigma": sigma,
        "mse_mean": mse_mean,
        "psnr_mean": float(psnr.mean().item()),
        "ssim_mean": ssim_mean,
    }
    (out_dir / "gmi_metrics.json").write_text(json.dumps(metrics, indent=2))

    return metrics


def _run_gmi_with_pretrained_gan(
    dataset: DatasetDescriptor,
    model_path: Path,
    sigma: float,
    batch_size: int,
    num_workers: int,
    preprocess_resolution: int,
    optimize_num: int,
    z_dim: int,
    inner_iter_times: int,
    class_loss_weight: float,
    disc_loss_weight: float,
    generator_ckpt_path: Path,
    discriminator_ckpt_path: Path,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = int(dataset.train_labels.max().item()) + 1

    target_model = _load_target_model_from_checkpoint(model_path, num_classes)
    generator = auto_generator_from_pretrained(str(generator_ckpt_path))
    discriminator = auto_discriminator_from_pretrained(str(discriminator_ckpt_path))

    target_model = target_model.to(device)
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    target_model.eval()
    generator.eval()
    discriminator.eval()

    latents_sampler = SimpleLatentsSampler(z_dim, batch_size)
    optimization_config = SimpleWhiteBoxOptimizationConfig(
        experiment_dir=str(model_path.parent / "gmi"),
        device=device,
        optimizer="SGD",
        optimizer_kwargs={"lr": 0.02, "momentum": 0.9},
        iter_times=inner_iter_times,
        show_loss_info_iters=max(1, inner_iter_times // 5),
    )

    identity_loss_fn = ImageAugmentClassificationLoss(
        classifier=target_model, loss_fn="ce", create_aug_images_fn=None
    )
    discriminator_loss_fn = GmiDiscriminatorLoss(discriminator)
    loss_fn = ComposeImageLoss(
        [identity_loss_fn, discriminator_loss_fn],
        weights=[class_loss_weight, disc_loss_weight],
    )

    optimization = SimpleWhiteBoxOptimization(optimization_config, generator, loss_fn)

    # Build evaluation dataset for FID/PRDC.
    pt_dir = dataset.root_dir / "dataset"
    pt_path = pt_dir / ("train_public.pt" if (pt_dir / "train_public.pt").exists() else "train.pt")
    eval_dataset = PTImageDataset(
        pt_path,
        dataset.name,
        preprocess_resolution,
        transform=transforms.ToTensor(),
    )

    subset_images = []
    subset_labels = []
    for cls in dataset.test_labels.unique():
        cls_indices = (dataset.test_labels == cls).nonzero(as_tuple=True)[0][:optimize_num]
        subset_images.append(dataset.test_images[cls_indices])
        subset_labels.append(dataset.test_labels[cls_indices])
    subset_images = torch.cat(subset_images, dim=0)
    subset_labels = torch.cat(subset_labels, dim=0)

    loader = DataLoader(
        TensorDataset(subset_images, subset_labels),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    reconstructions = []
    gt_images = []
    gt_labels = []

    for imgs, labels in tqdm(loader, leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)
        targets = F.interpolate(
            imgs, size=(preprocess_resolution, preprocess_resolution), mode="bilinear", align_corners=False
        )
        if dataset.in_channels == 1:
            targets = targets.repeat(1, 3, 1, 1)

        latents = latents_sampler(list(labels.cpu().numpy()), len(labels)).to(device)
        output = optimization(latents, labels)
        reconstructions.append(output.images.cpu())
        gt_images.append(targets.cpu())
        gt_labels.append(labels.cpu())

    reconstructions = torch.cat(reconstructions, dim=0)
    gt_images = torch.cat(gt_images, dim=0)
    gt_labels = torch.cat(gt_labels, dim=0)

    mse = F.mse_loss(reconstructions, gt_images, reduction="none").view(len(reconstructions), -1).mean(dim=1)
    mse_mean = mse.mean().item()
    psnr = 10 * torch.log10(1.0 / torch.clamp(mse, min=1e-8))

    try:
        from piq import ssim
    except Exception:
        def ssim(x, y):
            mu_x = x.mean(dim=(-2, -1))
            mu_y = y.mean(dim=(-2, -1))
            sigma_x = x.var(dim=(-2, -1))
            sigma_y = y.var(dim=(-2, -1))
            sigma_xy = ((x - mu_x[..., None, None]) * (y - mu_y[..., None, None])).var(dim=(-2, -1))
            c1 = 0.01 ** 2
            c2 = 0.03 ** 2
            ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
                (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
            )
            return ssim_map.mean(dim=1)
    ssim_scores = ssim(reconstructions, gt_images)
    if isinstance(ssim_scores, torch.Tensor):
        ssim_scores = ssim_scores.cpu()
    ssim_mean = float(torch.tensor(ssim_scores).mean().item())

    # FID/PRDC on reconstructed images
    fid_prdc_metric = ImageFidPRDCMetric(
        batch_size,
        eval_dataset,
        device=device,
        save_individual_prdc_dir=str(model_path.parent / "gmi"),
        fid=True,
        prdc=True,
    )
    features = fid_prdc_metric.get_features(reconstructions, gt_labels)
    fid_results = fid_prdc_metric(features, gt_labels)

    out_dir = model_path.parent / "gmi"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_image(make_grid(reconstructions, nrow=batch_size, normalize=True), out_dir / "reconstructions.png")

    metrics = {
        "dataset": dataset.name,
        "sigma": sigma,
        "mse_mean": mse_mean,
        "psnr_mean": float(psnr.mean().item()),
        "ssim_mean": ssim_mean,
    }
    metrics.update({k.lower(): float(v) for k, v in fid_results.items()})
    (out_dir / "gmi_metrics.json").write_text(json.dumps(metrics, indent=2))

    return metrics


def construct_generator():
    # simple 64x64 generator from repo
    from ..models.gans.simple import SimpleGenerator64

    generator = SimpleGenerator64()
    return generator


import gdown
