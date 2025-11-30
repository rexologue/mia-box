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
    format_sigma,
    load_datasets_from_pt,
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
    datasets = load_datasets_from_pt(
        args.exp_root, use_public=args.use_public_only, dataset_names=args.datasets
    )
    summary = {}

    for dataset in datasets:
        LOGGER.info("Running GMI for dataset %s", dataset.name)
        dataset_records = {}
        for sigma in args.noise_levels:
            sigma_str = format_sigma(sigma)
            model_path = dataset.root_dir / f"noise_{sigma_str}" / "model.pt"
            if not model_path.exists():
                LOGGER.warning("Missing checkpoint for sigma %s", sigma_str)
                continue
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
            )
            dataset_records[sigma] = metrics
        if dataset_records:
            for metric_name in ["mse_mean", "psnr_mean", "ssim_mean"]:
                plot_metric_curve(dataset.name, metric_name, dataset_records, dataset.root_dir)
            summary[dataset.name] = dataset_records

    (args.exp_root / "gmi_summary.json").write_text(json.dumps(summary, indent=2))
    LOGGER.info("GMI summary written to %s", args.exp_root / "gmi_summary.json")


if __name__ == "__main__":
    main()
