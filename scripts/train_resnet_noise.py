import argparse
import json
import logging
from pathlib import Path
from typing import List
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import matplotlib.pyplot as plt
import torch
from modelinversion.experiments.noise_utils import (
    ensure_datasets_pt,
    load_datasets_from_pt,
    train_resnet_for_dataset,
    format_sigma,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet-34 models with Gaussian noise")
    parser.add_argument("--exp_root", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--noise_levels",
        type=float,
        nargs="+",
        default=[x / 10 for x in range(0, 11)],
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["mnist", "fmnist", "overhead_mnist"],
        help="Datasets to include",
    )
    parser.add_argument("--use_public_only", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_prepare", action="store_true")
    return parser.parse_args()


def summarize_dataset_metrics(dataset_name: str, exp_root: Path, noise_levels: List[float]):
    xs = []
    accs = []
    losses = []
    for sigma in noise_levels:
        sigma_str = format_sigma(sigma)
        metrics_path = exp_root / dataset_name / f"noise_{sigma_str}" / "metrics.json"
        if not metrics_path.exists():
            continue
        data = json.loads(metrics_path.read_text())
        xs.append(sigma)
        accs.append(data.get("test_acc", 0))
        losses.append(data.get("min_train_loss", 0))
    if not xs:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(xs, accs, marker="o")
    ax[0].set_xlabel("sigma")
    ax[0].set_ylabel("test acc")
    ax[0].set_title("Accuracy vs noise")
    ax[1].plot(xs, losses, marker="o", color="orange")
    ax[1].set_xlabel("sigma")
    ax[1].set_ylabel("min train loss")
    ax[1].set_title("Loss vs noise")
    plt.tight_layout()
    fig.savefig(exp_root / dataset_name / "summary_sigma_curves.png", bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    if not args.skip_prepare:
        ensure_datasets_pt(
            args.exp_root, split_public_private=True, dataset_names=args.datasets
        )

    datasets = load_datasets_from_pt(
        args.exp_root, use_public=args.use_public_only, dataset_names=args.datasets
    )

    for dataset in datasets:
        LOGGER.info("Processing dataset %s", dataset.name)
        for sigma in args.noise_levels:
            train_resnet_for_dataset(
                dataset,
                sigma,
                batch_size=args.batch_size,
                epochs=args.epochs,
                num_workers=args.num_workers,
                learning_rate=args.learning_rate,
                seed=args.seed,
            )
        summarize_dataset_metrics(dataset.name, args.exp_root, args.noise_levels)


if __name__ == "__main__":
    main()
