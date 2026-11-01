# Uncertainty-Aware Point Cloud Segmentation (ANN based experiments)

This is the ANN codebase for the paper "Role-based Effectiveness of an Epistemic Uncertainty Estimate in Training Models for Semantic Segmentation of 3D LiDAR Point Clouds". Its advised to run the models either on Linux or WSL, otherwise library compatibility issues may arise.

Before using the notebooks, kindly set the associated directories and subfolders for the datasets in `datasets.yaml`

## Structure

```
configs/        # Hyperparameters and dataset config files
data/           # Dataset paths and helper functiosn
src/            # Entire reusable logic or modules
notebooks/      # Notebooks per pipeline stages
checkpoints/    # Model weights (ignored)
results/        # Inference files and results (ignored)
ProcessedData/  # Feature CSVs (ignored)
RawDatasets/    # Raw point cloud files (ignored)
```

## Setup

```bash
conda env create -f environment.yml
conda activate uncertainty-pc
pip install -e .
```

Set the `DATA_ROOT` environment variable (or edit `configs/datasets.yaml`) to point at the actual data drive.

```bash
export DATA_ROOT=/mnt/d/Krishna/Research/UncertaintyEstimation/Code/ANNs/NewCode
```

## Conventions

- **Splits are file-based / spatial, not random row splits.** Two types are there, `src/spatial_split.py` and the `canonical_split` block per dataset in `configs/datasets.yaml`.
- **Scalers/encoders are always fit on train-only data**, then applied to test (`src/normalization.py`).
