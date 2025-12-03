import argparse
import json
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import matplotlib.pyplot as plt
from modelinversion.experiments.noise_utils import (
    DEFAULT_GAN_DISCRIMINATOR_URL,
    DEFAULT_GAN_GENERATOR_URL,
    copy_gan_checkpoints_for_attack,
    ensure_gan_checkpoints,
    format_sigma,
    load_datasets_from_pt,
    resolve_gan_base_dir,
    require_gan_checkpoints,
    run_simple_gmi,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run GMI attacks across noise levels")
    parser.add_argument("--exp_root", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--samples_per_class", type=int, default=5)
    parser.add_argument("--iter_times", type=int, default=200)
    parser.add_argument("--mse_weight", type=float, default=1.0)
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument(
        "--noise_levels",
        type=float,
        nargs="+",
        default=[x / 10 for x in range(0, 11)],
    )
    parser.add_argument(
        "--gan_ckpt_dir",
        type=Path,
        default=None,
        help="Optional root for GAN checkpoints (defaults to <exp_root>/<dataset>/gan_base).",
    )
    parser.add_argument(
        "--gan_base_dir_template",
        type=str,
        default=None,
        help="Format string for GAN base directories using {dataset} and {exp_root}.",
    )
    parser.add_argument(
        "--gan_use_pretrained",
        action="store_true",
        help="Download or copy pretrained GANs instead of assuming they were trained in-pipeline.",
    )
    parser.add_argument(
        "--gan_pretrained_dir",
        type=Path,
        default=None,
        help="Directory containing pretrained GAN checkpoints to copy into the base directory.",
    )
    parser.add_argument(
        "--generator_url",
        type=str,
        default=DEFAULT_GAN_GENERATOR_URL,
        help="Google Drive URL for the base generator checkpoint.",
    )
    parser.add_argument(
        "--discriminator_url",
        type=str,
        default=DEFAULT_GAN_DISCRIMINATOR_URL,
        help="Google Drive URL for the base discriminator checkpoint.",
    )
    parser.add_argument("--preprocess_resolution", type=int, default=64)
    parser.add_argument("--optimize_num", type=int, default=50)
    parser.add_argument("--inner_iter_times", type=int, default=1500)
    parser.add_argument("--class_loss_weight", type=float, default=100.0)
    parser.add_argument("--disc_loss_weight", type=float, default=1.0)
    parser.add_argument(
        "--save_image_iters",
        type=int,
        nargs="+",
        default=[50, 100, 150, 200],
        help="Iterations within each optimization run at which to save reconstruction grids.",
    )
    parser.add_argument("--use_public_only", action="store_true")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["mnist", "fmnist", "overhead_mnist"],
    )
    return parser.parse_args()


def plot_metric_curve(dataset_name: str, metric_name: str, records: dict, root: Path):
    xs = sorted(records.keys())
    ys = [records[x][metric_name] for x in xs]
    plt.figure(figsize=(5, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("sigma")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} vs noise ({dataset_name})")
    plt.tight_layout()
    plt.savefig(root / f"gmi_{metric_name}_vs_noise.png", bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    base_dir_template = args.gan_base_dir_template

    def _resolve_base_dir(dataset_name: str) -> Path:
        if base_dir_template:
            return Path(base_dir_template.format(dataset=dataset_name, exp_root=args.exp_root))
        if args.gan_ckpt_dir:
            return Path(args.gan_ckpt_dir) / dataset_name / "gan_base"
        return resolve_gan_base_dir(args.exp_root, dataset_name)

    datasets = load_datasets_from_pt(
        args.exp_root, use_public=args.use_public_only, dataset_names=args.datasets
    )
    summary = {}

    for dataset in datasets:
        LOGGER.info("Running GMI for dataset %s", dataset.name)
        base_dir = _resolve_base_dir(dataset.name)
        if args.gan_use_pretrained:
            base_generator_ckpt, base_discriminator_ckpt = ensure_gan_checkpoints(
                base_dir,
                generator_url=args.generator_url,
                discriminator_url=args.discriminator_url,
                pretrained_dir=args.gan_pretrained_dir,
            )
        else:
            base_generator_ckpt, base_discriminator_ckpt = require_gan_checkpoints(base_dir)

        dataset_records = {}
        for sigma in args.noise_levels:
            sigma_str = format_sigma(sigma)
            model_path = dataset.root_dir / f"noise_{sigma_str}" / "model.pt"
            if not model_path.exists():
                LOGGER.warning("Missing checkpoint for sigma %s", sigma_str)
                continue
            generator_ckpt, discriminator_ckpt = copy_gan_checkpoints_for_attack(
                model_path.parent, base_generator_ckpt, base_discriminator_ckpt
            )
            metrics = run_simple_gmi(
                dataset,
                model_path,
                sigma,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                samples_per_class=args.samples_per_class,
                mse_weight=args.mse_weight,
                ce_weight=args.ce_weight,
                iter_times=args.iter_times,
                save_image_iters=[
                    i for i in args.save_image_iters if 0 < i <= args.inner_iter_times
                ],
                generator_ckpt_path=generator_ckpt,
                discriminator_ckpt_path=discriminator_ckpt,
                preprocess_resolution=args.preprocess_resolution,
                optimize_num=args.optimize_num,
                z_dim=100,
                inner_iter_times=args.inner_iter_times,
                class_loss_weight=args.class_loss_weight,
                disc_loss_weight=args.disc_loss_weight,
            )
            dataset_records[sigma] = metrics
        if dataset_records:
            available_metrics = set().union(*(m.keys() for m in dataset_records.values()))
            metric_names = ["mse_mean", "psnr_mean", "ssim_mean"]
            if "fid" in available_metrics:
                metric_names.append("fid")
            for metric_name in metric_names:
                plot_metric_curve(dataset.name, metric_name, dataset_records, dataset.root_dir)
            summary[dataset.name] = dataset_records

    (args.exp_root / "gmi_summary.json").write_text(json.dumps(summary, indent=2))
    LOGGER.info("GMI summary written to %s", args.exp_root / "gmi_summary.json")


if __name__ == "__main__":
    main()
