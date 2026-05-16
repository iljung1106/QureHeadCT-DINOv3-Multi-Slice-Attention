from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ctmil.dicom_utils import read_dicom_meta
from ctmil.labels import normalize_case_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom-root", default="data/raw/qureai-headct")
    parser.add_argument("--out-csv", default="data/processed/dicom_index.csv")
    args = parser.parse_args()

    root = Path(args.dicom_root)
    paths = sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {"", ".dcm"}])
    rows = []
    for path in tqdm(paths, desc="Indexing DICOM"):
        try:
            rows.append(read_dicom_meta(path).__dict__)
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No DICOM files indexed under {root}")
    df["case_id"] = df["patient_id"].map(normalize_case_id)
    df = df.sort_values(["case_id", "study_uid", "series_uid", "image_position_z", "instance_number"])
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(df)} slices from {df['case_id'].nunique()} cases to {args.out_csv}")


if __name__ == "__main__":
    main()
