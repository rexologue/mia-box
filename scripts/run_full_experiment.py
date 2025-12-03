import argparse
import logging
from pathlib import Path
from typing import Any, Dict
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from modelinversion.experiments.noise_utils import (
    DEFAULT_GAN_DISCRIMINATOR_URL,
    DEFAULT_GAN_GENERATOR_URL,
    ensure_datasets_pt,
    load_datasets_from_pt,
    prepare_gan_checkpoints_for_datasets,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)


DEFAULT_STEPS = {
    "prepare_datasets": True,
    "train_gan": True,
    "train_models": True,
    "eval_classification": True,
    "run_gmi": True,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run full noise/GMI experiment")
    parser.add_argument("--config", type=Path, required=True, help="YAML config file")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data


def run_prepare(exp_root: Path, dataset_names):
    ensure_datasets_pt(exp_root, split_public_private=True, dataset_names=dataset_names)
    LOGGER.info("Datasets prepared at %s", exp_root)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    exp_root = Path(cfg.get("exp_root", "experiments/gmi_noise"))
    dataset_names = cfg.get("datasets", ["mnist", "fmnist", "overhead_mnist"])
    steps = DEFAULT_STEPS | cfg.get("steps", {})
    gan_cfg = cfg.get("gan", {})

    # set seeds
    seed = cfg.get("training", {}).get("seed", 42)
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if steps.get("prepare_datasets", True):
        run_prepare(exp_root, dataset_names)

    if steps.get("train_gan", True) and gan_cfg.get("enabled", True):
        prepare_gan_checkpoints_for_datasets(exp_root, dataset_names, gan_cfg)

    if steps.get("train_models", True):
        # Build args for training script
        train_args = [
            "--exp_root", str(exp_root),
            "--batch_size", str(cfg.get("training", {}).get("batch_size", 64)),
            "--epochs", str(cfg.get("training", {}).get("num_epochs", 3)),
            "--num_workers", str(cfg.get("training", {}).get("num_workers", 4)),
            "--learning_rate", str(cfg.get("training", {}).get("learning_rate", 1e-3)),
            "--seed", str(seed),
            "--datasets", *dataset_names,
        ]
        if cfg.get("use_public_only", True):
            train_args.append("--use_public_only")
        noise_levels = cfg.get("noise_levels")
        if noise_levels:
            train_args.extend(["--noise_levels", *[str(x) for x in noise_levels]])
        sys.argv = ["train_resnet_noise.py", *train_args]
        train_module = __import__("train_resnet_noise")
        train_module.main()

    if steps.get("eval_classification", True):
        eval_args = [
            "--exp_root", str(exp_root),
            "--batch_size", str(cfg.get("training", {}).get("batch_size", 64)),
            "--num_workers", str(cfg.get("training", {}).get("num_workers", 4)),
            "--datasets", *dataset_names,
        ]
        if cfg.get("use_public_only", True):
            eval_args.append("--use_public_only")
        noise_levels = cfg.get("noise_levels")
        if noise_levels:
            eval_args.extend(["--noise_levels", *[str(x) for x in noise_levels]])
        sys.argv = ["eval_classification_metrics.py", *eval_args]
        eval_module = __import__("eval_classification_metrics")
        eval_module.main()

    if steps.get("run_gmi", False) and cfg.get("gmi", {}).get("enabled", True):
        gmi_cfg = cfg.get("gmi", {})
        gan_root = gan_cfg.get("gan_ckpt_dir") or gan_cfg.get("base_dir_root")
        gan_base_template = gan_cfg.get("base_dir_template")
        gan_pretrained_cfg = gan_cfg.get("pretrained", {}) if isinstance(gan_cfg, dict) else {}
        gmi_args = [
            "--exp_root", str(exp_root),
            "--batch_size", str(gmi_cfg.get("batch_size", 16)),
            "--num_workers", str(gmi_cfg.get("num_workers", 4)),
            "--samples_per_class", str(gmi_cfg.get("samples_per_class", 5)),
            "--iter_times", str(gmi_cfg.get("iter_times", 200)),
            "--mse_weight", str(gmi_cfg.get("mse_weight", 1.0)),
            "--ce_weight", str(gmi_cfg.get("ce_weight", 1.0)),
            "--datasets", *dataset_names,
        ]
        if gan_root:
            gmi_args.extend(["--gan_ckpt_dir", str(gan_root)])
        if gan_base_template:
            gmi_args.extend(["--gan_base_dir_template", gan_base_template])
        if gan_cfg.get("use_pretrained", False):
            gmi_args.append("--gan_use_pretrained")
        if gan_pretrained_cfg.get("directory"):
            gmi_args.extend(["--gan_pretrained_dir", str(gan_pretrained_cfg["directory"])])
        gmi_args.extend(
            [
                "--generator_url",
                gan_pretrained_cfg.get("generator_url", DEFAULT_GAN_GENERATOR_URL),
                "--discriminator_url",
                gan_pretrained_cfg.get(
                    "discriminator_url", DEFAULT_GAN_DISCRIMINATOR_URL
                ),
            ]
        )
        if cfg.get("use_public_only", True):
            gmi_args.append("--use_public_only")
        noise_levels = cfg.get("noise_levels")
        if noise_levels:
            gmi_args.extend(["--noise_levels", *[str(x) for x in noise_levels]])
        sys.argv = ["run_gmi_experiments.py", *gmi_args]
        gmi_module = __import__("run_gmi_experiments")
        gmi_module.main()

    LOGGER.info("Full experiment finished. Outputs saved under %s", exp_root)


if __name__ == "__main__":
    main()
