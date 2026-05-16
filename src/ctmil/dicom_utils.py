from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydicom


@dataclass(frozen=True)
class DicomMeta:
    path: str
    patient_id: str
    study_uid: str
    series_uid: str
    instance_number: float
    slice_location: float
    image_position_z: float


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, (list, tuple)):
            return float(value[-1])
        return float(value)
    except Exception:
        return default


def read_dicom_meta(path: str | Path) -> DicomMeta:
    ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    image_position = getattr(ds, "ImagePositionPatient", [0, 0, 0])
    return DicomMeta(
        path=str(path),
        patient_id=str(getattr(ds, "PatientID", Path(path).parts[-3] if len(Path(path).parts) >= 3 else "")),
        study_uid=str(getattr(ds, "StudyInstanceUID", "")),
        series_uid=str(getattr(ds, "SeriesInstanceUID", "")),
        instance_number=_as_float(getattr(ds, "InstanceNumber", 0)),
        slice_location=_as_float(getattr(ds, "SliceLocation", 0)),
        image_position_z=_as_float(image_position, 0),
    )


def load_hu(path: str | Path) -> np.ndarray:
    ds = pydicom.dcmread(str(path), force=True)
    image = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return image * slope + intercept


def window_to_uint8(hu: np.ndarray, center: float, width: float) -> np.ndarray:
    low = center - width / 2
    high = center + width / 2
    clipped = np.clip(hu, low, high)
    scaled = (clipped - low) / max(high - low, 1e-6)
    return (scaled * 255).astype(np.uint8)


def ct_to_rgb_windows(
    hu: np.ndarray,
    windows: tuple[tuple[float, float], ...] = ((40, 80), (80, 200), (600, 2800)),
) -> np.ndarray:
    channels = [window_to_uint8(hu, center, width) for center, width in windows]
    return np.stack(channels, axis=-1)

