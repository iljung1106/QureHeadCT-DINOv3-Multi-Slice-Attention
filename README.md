# DINOv3 Head CT MIL

This project builds a frozen baseline for Head CT scan classification:

```text
DINOv3 slice encoder -> BiGRU sequence encoder -> Gated ABMIL -> multi-label scan classifier
```

The frozen baseline trains only the BiGRU, attention pooling, and classifier on cached DINOv3
slice features. It is the comparison point for later LoRA fine-tuning of DINOv3.

## Setup

Use Python 3.10-3.12 for PyTorch compatibility.

```bash
bash scripts/setup_venv.sh
source .venv/bin/activate
```

Copy `.env.example` to `.env` or keep the existing `.env.txt`. Supported variables:

```text
HF_KEY=...
KAGGLE_USERNAME=...
KAGGLE_KEY=...
```

## Download

```bash
python scripts/download_hf_model.py
python scripts/download_kaggle_dataset.py --unzip
```

Kaggle download requires accepted dataset terms and Kaggle API credentials.

## Data Preparation

```bash
python scripts/build_fast_path_index.py --dicom-root data/raw/qureai-headct
python scripts/normalize_labels.py --labels-csv path/to/label_file.csv --id-column PatientID
python scripts/make_patient_split.py
```

The split file is case-level and is saved to `splits/cq500_seed42.csv`.

## Feature Extraction

```bash
python scripts/extract_dinov3_features.py --split-csv splits/cq500_seed42.csv --split train
python scripts/extract_dinov3_features.py --split-csv splits/cq500_seed42.csv --split val
python scripts/extract_dinov3_features.py --split-csv splits/cq500_seed42.csv --split test
```

For a CPU smoke test:

```bash
python scripts/extract_dinov3_features.py --model data/raw/hf/dinov3-vitb16-pretrain-lvd1689m --split-csv splits/cq500_seed42.csv --split train --limit-cases 2 --max-slices 2 --batch-size 1
```

## Frozen Baseline

```bash
python scripts/train_frozen_baseline.py --config configs/baseline.yaml
```

Outputs:

- `data/models/frozen_bigru_abmil/best.pt`
- `data/models/frozen_bigru_abmil/history.csv`
- `data/models/frozen_bigru_abmil/metrics.json`
- `data/models/frozen_bigru_abmil/test_predictions.csv`

## Vast.ai / GPU Notebook

For a GPU host such as Vast.ai, start with:

- `notebooks/01_vast_frozen_baseline.ipynb`
- `docs/vast_ai_runbook.md`

The one-command shell equivalent is:

```bash
bash scripts/run_frozen_baseline_vast.sh
```

## Slice-Level Frozen Head

Seg-CQ500 provides 3D hemorrhage segmentation masks for 51 CQ500 scans. Build slice labels from
those masks with:

```bash
python scripts/download_seg_cq500.py --unzip
python scripts/build_seg_cq500_slice_labels.py \
  --seg-root data/raw/seg-cq500 \
  --index-csv data/processed/dicom_index_fast.csv \
  --out-csv data/processed/slice_labels.csv
```

This creates:

```text
data/processed/slice_labels.csv
```

with columns:

```text
case_id, hemorrhage, path
```

Then run:

```bash
python scripts/train_slice_head.py --config configs/vast/slice_head.yaml
```

This trains only a small binary head on cached frozen DINOv3 slice features. It does not train
DINOv3 and it is independent of the BiGRU-GatedABMIL scan classifier.
