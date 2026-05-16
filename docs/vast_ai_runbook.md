# Vast.ai Runbook

This runbook is for running the DINOv3 frozen baseline on a GPU instance.

## Recommended Instance

- CUDA-capable NVIDIA GPU
- 12 GB VRAM minimum, 16-24 GB preferred
- 100 GB disk minimum if downloading CQ500 on the instance
- PyTorch image with CUDA, or Ubuntu image with Python 3.10-3.12

## Environment Variables

Create `.env` in the repository root:

```text
HF_KEY=...
KAGGLE_USERNAME=...
KAGGLE_KEY=...
```

The Kaggle account must have accepted the Qure.ai HeadCT dataset terms.

## Fast Path

```bash
git clone <your-repo-url> dinov3-ct-mil
cd dinov3-ct-mil

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

python scripts/download_hf_model.py
python scripts/download_kaggle_dataset.py --unzip
python scripts/normalize_labels.py --labels-csv data/raw/qureai-headct/reads.csv --id-column name
python scripts/build_fast_path_index.py --dicom-root data/raw/qureai-headct --out-csv data/processed/dicom_index_fast.csv
python scripts/filter_matched_labels.py
python scripts/make_patient_split.py --labels-csv data/processed/labels_matched.csv --out-csv splits/cq500_seed42.csv

python scripts/extract_dinov3_features.py --model data/raw/hf/dinov3-vitb16-pretrain-lvd1689m --index-csv data/processed/dicom_index_fast.csv --split-csv splits/cq500_seed42.csv --split train --batch-size 32
python scripts/extract_dinov3_features.py --model data/raw/hf/dinov3-vitb16-pretrain-lvd1689m --index-csv data/processed/dicom_index_fast.csv --split-csv splits/cq500_seed42.csv --split val --batch-size 32
python scripts/extract_dinov3_features.py --model data/raw/hf/dinov3-vitb16-pretrain-lvd1689m --index-csv data/processed/dicom_index_fast.csv --split-csv splits/cq500_seed42.csv --split test --batch-size 32

python scripts/train_frozen_baseline.py --config configs/vast/frozen_baseline.yaml
```

## Notes

- If VRAM is low, reduce feature extraction `--batch-size` to 8 or 16.
- Frozen baseline training uses cached features and is much cheaper than DINOv3 feature extraction.
- The current baseline is scan-level only. LoRA and slice-level auxiliary training should be added as a second notebook after the frozen baseline is reproducible.

