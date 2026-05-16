#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python scripts/download_hf_model.py
python scripts/download_kaggle_dataset.py --unzip
python scripts/normalize_labels.py --labels-csv data/raw/qureai-headct/reads.csv --id-column name
python scripts/build_fast_path_index.py --dicom-root data/raw/qureai-headct --out-csv data/processed/dicom_index_fast.csv
python scripts/filter_matched_labels.py
python scripts/download_seg_cq500.py --unzip
python scripts/build_seg_cq500_slice_labels.py --seg-root data/raw/seg-cq500 --case-list-only
python scripts/make_patient_split.py \
  --labels-csv data/processed/labels_matched.csv \
  --out-csv splits/cq500_seed42.csv \
  --fixed-cases-csv data/processed/seg_cq500_cases.csv \
  --fixed-train-count 34 \
  --fixed-val-count 5 \
  --fixed-test-count 10

MODEL_DIR="data/raw/hf/dinov3-vitb16-pretrain-lvd1689m"
for SPLIT in train val test; do
  python scripts/extract_dinov3_features.py \
    --model "$MODEL_DIR" \
    --index-csv data/processed/dicom_index_fast.csv \
    --split-csv splits/cq500_seed42.csv \
    --split "$SPLIT" \
    --batch-size "${FEATURE_BATCH_SIZE:-32}" \
    --out-dir data/features/dinov3_vitb16
done

python scripts/train_frozen_baseline.py --config configs/vast/frozen_baseline.yaml

if [[ "${RUN_SEG_CQ500_SLICE_HEAD:-0}" == "1" ]]; then
  python scripts/build_seg_cq500_slice_labels.py \
    --seg-root data/raw/seg-cq500 \
    --index-csv data/processed/dicom_index_fast.csv \
    --out-csv data/processed/slice_labels.csv
  python scripts/train_slice_head.py --config configs/vast/slice_head.yaml
fi

if [[ "${RUN_LORA_MIL:-0}" == "1" ]]; then
  python scripts/train_lora_mil.py --config configs/vast/lora_mil.yaml
fi
