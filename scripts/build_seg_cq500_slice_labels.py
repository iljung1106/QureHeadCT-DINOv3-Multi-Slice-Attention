from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ctmil.labels import normalize_case_id


MASK_SUFFIXES = (".nii.gz", ".nii", ".nrrd", ".mha", ".mhd")


def mask_suffix(path: Path) -> str:
    name = path.name.lower()
    for suffix in MASK_SUFFIXES:
        if name.endswith(suffix):
            return suffix
    return ""


def find_mask_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        if path.is_file() and mask_suffix(path):
            files.append(path)
    return sorted(files)


def load_mask(path: Path) -> np.ndarray:
    suffix = mask_suffix(path)
    if suffix in {".nii", ".nii.gz"}:
        import nibabel as nib

        return np.asarray(nib.load(str(path)).get_fdata())
    if suffix in {".nrrd", ".mha", ".mhd"}:
        import SimpleITK as sitk

        return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    raise ValueError(f"Unsupported mask file: {path}")


def infer_case_id(path: Path) -> str:
    text = " ".join(path.parts)
    match = re.search(r"CQ500[-_\s]*CT[-_\s]*0*(\d+)|CQ500CT0*(\d+)", text, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not infer CQ500 case id from {path}")
    number = match.group(1) or match.group(2)
    return normalize_case_id(f"CQ500CT{number}")


def labels_along_axis(mask: np.ndarray, axis: int) -> np.ndarray:
    reduce_axes = tuple(i for i in range(mask.ndim) if i != axis)
    return (np.asarray(mask) > 0).any(axis=reduce_axes).astype(int)


def choose_axis_and_series(mask: np.ndarray, case_index: pd.DataFrame) -> tuple[int, str, pd.DataFrame, np.ndarray]:
    best = None
    for axis in range(mask.ndim):
        labels = labels_along_axis(mask, axis)
        for series_uid, series_df in case_index.groupby("series_uid"):
            diff = abs(len(series_df) - len(labels))
            candidate = (diff, axis, str(series_uid), series_df.sort_values("instance_number"), labels)
            if best is None or candidate[0] < best[0]:
                best = candidate
    if best is None:
        raise ValueError("No candidate DICOM series found")
    _, axis, series_uid, series_df, labels = best
    if len(series_df) != len(labels):
        n = min(len(series_df), len(labels))
        series_df = series_df.iloc[:n].copy()
        labels = labels[:n]
    return axis, series_uid, series_df, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seg-root", default="data/raw/seg-cq500")
    parser.add_argument("--index-csv", default="data/processed/dicom_index_fast.csv")
    parser.add_argument("--out-csv", default="data/processed/slice_labels.csv")
    parser.add_argument("--report-json", default="data/reports/seg_cq500_slice_label_report.json")
    args = parser.parse_args()

    seg_root = Path(args.seg_root)
    masks = find_mask_files(seg_root)
    if not masks:
        raise SystemExit(f"No mask files found under {seg_root}")
    index = pd.read_csv(args.index_csv)
    index["case_id"] = index["case_id"].map(normalize_case_id)

    rows = []
    report = {"masks_found": len(masks), "matched": [], "unmatched": []}
    for mask_path in masks:
        try:
            case_id = infer_case_id(mask_path)
            case_index = index[index["case_id"] == case_id].copy()
            if case_index.empty:
                report["unmatched"].append({"mask": str(mask_path), "reason": "case_not_in_dicom_index"})
                continue
            mask = load_mask(mask_path)
            if mask.ndim < 3:
                report["unmatched"].append({"mask": str(mask_path), "reason": f"mask_ndim_{mask.ndim}"})
                continue
            axis, series_uid, series_df, slice_labels = choose_axis_and_series(mask, case_index)
            for (_, row), label in zip(series_df.iterrows(), slice_labels):
                rows.append(
                    {
                        "case_id": case_id,
                        "path": row["path"],
                        "instance_number": int(row["instance_number"]),
                        "series_uid": series_uid,
                        "hemorrhage": int(label),
                        "source_mask": str(mask_path),
                        "mask_axis": int(axis),
                    }
                )
            report["matched"].append(
                {
                    "case_id": case_id,
                    "mask": str(mask_path),
                    "mask_shape": list(mask.shape),
                    "mask_axis": int(axis),
                    "series_uid": series_uid,
                    "slices": int(len(series_df)),
                    "positive_slices": int(np.sum(slice_labels)),
                }
            )
        except Exception as exc:
            report["unmatched"].append({"mask": str(mask_path), "reason": repr(exc)})

    if not rows:
        raise SystemExit("No slice labels were created. Inspect the report for mask/index mismatches.")
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} slice labels from {len(report['matched'])} masks to {out_csv}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()

