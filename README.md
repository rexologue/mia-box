# Noise-Robustness Experiments for ResNet-34 and GMI

This repository now includes a complete, reproducible pipeline to study how input noise affects classification accuracy and Generative Model Inversion (GMI) quality on MNIST, FashionMNIST, and Overhead-MNIST (downloaded from Kaggle).

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Directory layout

Key entrypoints live under `scripts/` and reusable helpers under `src/modelinversion/experiments/`. All experiment artefacts (datasets, checkpoints, metrics, plots) are written inside the configured `exp_root` (default `experiments/gmi_noise`).

## End-to-end run

```bash
python scripts/run_full_experiment.py --config configs/gmi_noise_experiment.yaml
```

The orchestrator will:

1. Download datasets and build deterministic public/private splits (`scripts/prepare_datasets.py`).
2. Train ResNet-34 classifiers for each noise level (`scripts/train_resnet_noise.py`).
3. Re-evaluate saved checkpoints on clean data (`scripts/eval_classification_metrics.py`).
4. Run GMI attacks for every checkpoint and noise level (`scripts/run_gmi_experiments.py`).

## Useful standalone commands

Prepare datasets only:

```bash
python scripts/prepare_datasets.py --exp_root experiments/gmi_noise
```

Train models (noise levels configurable):

```bash
python scripts/train_resnet_noise.py --exp_root experiments/gmi_noise --noise_levels 0 0.2 0.5
```

Evaluate saved checkpoints:

```bash
python scripts/eval_classification_metrics.py --exp_root experiments/gmi_noise
```

Run GMI sweeps:

```bash
python scripts/run_gmi_experiments.py --exp_root experiments/gmi_noise --iter_times 150
```

## Outputs

Per dataset and noise level (e.g., `experiments/gmi_noise/mnist/noise_0.3/`):

- `model.pt` – ResNet-34 checkpoint and metadata.
- `metrics.json` / `eval_metrics.json` – training and evaluation stats.
- `conf_matrix.png`, `train_curves.png` – visuals for accuracy and learning curves.
- `gmi/reconstructions.png`, `gmi_metrics.json` – inversion examples and similarity metrics.

Dataset-level summaries live alongside each dataset directory (`summary_sigma_curves.png`, `accuracy_vs_noise.png`, `gmi_*_vs_noise.png`), and top-level summaries are written to `classification_summary.json` and `gmi_summary.json`.

## Configuration

The sample configuration at `configs/gmi_noise_experiment.yaml` shows all tunable fields, including noise levels, batch sizes, epochs, and GMI hyperparameters. Adjust it to match your hardware budget or research focus.

## Model and attack components

- Classifiers: torchvision ResNet-34 wrapped via the repository classifier factory.
- Noise injection: configurable Gaussian noise added to training inputs.
- GMI: the built-in white-box optimization stack with the repository’s simple GAN generator, combining classification and pixel fidelity losses.

Run experiments on GPU if available (`torch.cuda.is_available()` is checked automatically); otherwise, CPU execution is supported for reproducibility.
