from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class FeatureBagDataset(Dataset):
    def __init__(
        self,
        labels_csv: str | Path,
        split_csv: str | Path,
        feature_dir: str | Path,
        split: str,
        label_cols: list[str],
    ) -> None:
        labels = pd.read_csv(labels_csv)
        splits = pd.read_csv(split_csv)
        df = labels.merge(splits[["case_id", "split"]], on="case_id", how="inner")
        df = df[df["split"] == split].reset_index(drop=True)
        self.df = df
        self.feature_dir = Path(feature_dir)
        self.label_cols = label_cols

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.df.iloc[idx]
        case_id = str(row["case_id"])
        path = self.feature_dir / f"{case_id}.pt"
        payload = torch.load(path, map_location="cpu")
        features = payload["features"].float() if isinstance(payload, dict) else payload.float()
        labels = torch.tensor(row[self.label_cols].astype(float).to_numpy(), dtype=torch.float32)
        return {"case_id": case_id, "features": features, "labels": labels}


def collate_feature_bags(batch: list[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
    lengths = torch.tensor([item["features"].shape[0] for item in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    feat_dim = int(batch[0]["features"].shape[-1])
    features = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    labels = torch.stack([item["labels"] for item in batch])
    case_ids: list[str] = []
    for i, item in enumerate(batch):
        feat = item["features"]
        n = feat.shape[0]
        features[i, :n] = feat
        mask[i, :n] = True
        case_ids.append(str(item["case_id"]))
    return {"case_id": case_ids, "features": features, "mask": mask, "lengths": lengths, "labels": labels}

