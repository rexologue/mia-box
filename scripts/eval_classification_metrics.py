import argparse
import json
import logging
from pathlib import Path
from typing import Dict
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import matplotlib.pyplot as plt
import numpy as np
from modelinversion.experiments.noise_utils import (
    evaluate_checkpoint,
    format_sigma,
    load_datasets_from_pt,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved classifiers across noise levels")
    parser.add_argument("--exp_root", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
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


def save_summary_plot(dataset_name: str, metrics: Dict[float, float], out_dir: Path):
    xs = sorted(metrics.keys())
    ys = [metrics[x] for x in xs]
    plt.figure(figsize=(5, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("sigma")
    plt.ylabel("accuracy")
    plt.title(f"Accuracy vs noise ({dataset_name})")
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_vs_noise.png", bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    datasets = load_datasets_from_pt(
        args.exp_root, use_public=args.use_public_only, dataset_names=args.datasets
    )
    exp_summary = {}

    for dataset in datasets:
        LOGGER.info("Evaluating dataset %s", dataset.name)
        dataset_summary = {}
        for sigma in args.noise_levels:
            sigma_str = format_sigma(sigma)
            model_path = dataset.root_dir / f"noise_{sigma_str}" / "model.pt"
            if not model_path.exists():
                LOGGER.warning("Missing model for sigma %s in %s", sigma_str, dataset.name)
                continue
            acc, cm = evaluate_checkpoint(model_path, dataset, args.batch_size, args.num_workers)
            dataset_summary[sigma] = acc
            out_dir = model_path.parent
            np.save(out_dir / "conf_matrix_eval.npy", cm)
            (out_dir / "eval_metrics.json").write_text(
                json.dumps({"sigma": sigma, "test_acc": acc}, indent=2)
            )
        if dataset_summary:
            save_summary_plot(dataset.name, dataset_summary, dataset.root_dir)
            exp_summary[dataset.name] = dataset_summary

    (args.exp_root / "classification_summary.json").write_text(
        json.dumps(exp_summary, indent=2)
    )
    LOGGER.info("Done. Summary saved to %s", args.exp_root / "classification_summary.json")


if __name__ == "__main__":
    main()
