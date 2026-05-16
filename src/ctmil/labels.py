from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


DEFAULT_LABEL_ALIASES = {
    "ICH": ["ich", "intracranial", "hemorrhage", "haemorrhage"],
    "IPH": ["iph", "intraparenchymal"],
    "IVH": ["ivh", "intraventricular"],
    "SDH": ["sdh", "subdural"],
    "EDH": ["edh", "extradural", "epidural"],
    "SAH": ["sah", "subarachnoid"],
    "MidlineShift": ["midline", "shift"],
    "MassEffect": ["mass", "effect"],
}


def normalize_case_id(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9]", "", str(value)).upper()
    match = re.search(r"CQ500CT0*(\d+)", text)
    if match:
        return f"CQ500CT{int(match.group(1))}"
    return text


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [re.sub(r"\s+", "_", str(c).strip()) for c in out.columns]
    return out


def find_id_column(df: pd.DataFrame) -> str:
    candidates = ["patient_id", "PatientID", "Patient_Id", "study_id", "StudyInstanceUID", "SeriesInstanceUID"]
    for col in candidates:
        if col in df.columns:
            return col
    lowered = {c.lower(): c for c in df.columns}
    for key in ("patientid", "patient_id", "studyinstanceuid", "seriesinstanceuid", "study_id"):
        if key in lowered:
            return lowered[key]
    raise ValueError("Could not infer ID column. Pass a labels CSV with patient/study/series identifier.")


def _looks_like_label_column(column: str, aliases: list[str]) -> bool:
    normalized = column.lower().replace("_", " ")
    return all(alias in normalized for alias in aliases[:1]) or any(alias in normalized for alias in aliases)


def normalize_label_csv(
    labels_csv: str | Path,
    output_csv: str | Path,
    labels: list[str],
    id_column: str | None = None,
) -> None:
    df = canonicalize_columns(pd.read_csv(labels_csv))
    id_column = id_column or find_id_column(df)
    out = pd.DataFrame({"case_id": df[id_column].map(normalize_case_id)})
    for label in labels:
        aliases = DEFAULT_LABEL_ALIASES.get(label, [label])
        reader_matches = [
            col for col in df.columns if re.match(r"R\d+:", col) and _looks_like_label_column(col, aliases)
        ]
        matches = reader_matches or [col for col in df.columns if _looks_like_label_column(col, aliases)]
        if label in df.columns:
            matches = [label]
        if not matches:
            out[label] = 0
            continue
        values = df[matches].apply(pd.to_numeric, errors="coerce").fillna(0)
        if reader_matches:
            out[label] = (values.sum(axis=1) >= 2).astype(int)
        else:
            out[label] = (values.max(axis=1) > 0).astype(int)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.drop_duplicates("case_id").to_csv(output_csv, index=False)
