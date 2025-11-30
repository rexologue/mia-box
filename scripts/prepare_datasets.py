import argparse
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from modelinversion.experiments.noise_utils import ensure_datasets_pt

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare datasets and public/private splits")
    parser.add_argument("--exp_root", type=Path, required=True, help="Experiment root directory")
    parser.add_argument(
        "--skip_public_split", action="store_true", help="Do not create public/private splits"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["mnist", "fmnist", "overhead_mnist"],
        help="Datasets to prepare",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_datasets_pt(
        args.exp_root,
        split_public_private=not args.skip_public_split,
        dataset_names=args.datasets,
    )
    LOGGER.info("Datasets ready under %s", args.exp_root)


if __name__ == "__main__":
    main()
