from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-csv", default="data/processed/labels.csv")
    parser.add_argument("--index-csv", default="data/processed/dicom_index_fast.csv")
    parser.add_argument("--out-csv", default="data/processed/labels_matched.csv")
    parser.add_argument("--missing-csv", default="data/reports/labels_without_dicoms.csv")
    args = parser.parse_args()

    labels = pd.read_csv(args.labels_csv)
    index = pd.read_csv(args.index_csv, usecols=["case_id"])
    keep = set(index["case_id"].astype(str).unique())
    matched = labels[labels["case_id"].astype(str).isin(keep)].copy()
    missing = labels[~labels["case_id"].astype(str).isin(keep)].copy()

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.missing_csv).parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(args.out_csv, index=False)
    missing.to_csv(args.missing_csv, index=False)
    print(f"matched={len(matched)} missing={len(missing)}")


if __name__ == "__main__":
    main()

