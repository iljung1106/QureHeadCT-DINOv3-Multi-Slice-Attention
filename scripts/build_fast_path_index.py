from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ctmil.labels import normalize_case_id


def parse_instance(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom-root", default="data/raw/qureai-headct")
    parser.add_argument("--out-csv", default="data/processed/dicom_index.csv")
    args = parser.parse_args()

    root = Path(args.dicom_root)
    paths = sorted(root.rglob("*.dcm"))
    rows = []
    for path in tqdm(paths, desc="Indexing paths"):
        rel = path.relative_to(root)
        parts = rel.parts
        if len(parts) < 5:
            continue
        case_name = parts[1].split()[0]
        rows.append(
            {
                "path": str(path),
                "case_id": normalize_case_id(case_name),
                "patient_id": case_name,
                "study_uid": parts[2],
                "series_uid": parts[3],
                "instance_number": parse_instance(path),
                "slice_location": 0.0,
                "image_position_z": float(parse_instance(path)),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No DICOM paths indexed under {root}")
    df = df.sort_values(["case_id", "study_uid", "series_uid", "instance_number"])
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(df)} slices from {df['case_id'].nunique()} cases to {args.out_csv}")


if __name__ == "__main__":
    main()

