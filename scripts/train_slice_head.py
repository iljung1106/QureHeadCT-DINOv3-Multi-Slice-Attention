from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ctmil.labels import normalize_case_id
from ctmil.metrics import binary_metrics
from ctmil.train_utils import device_auto, load_yaml, save_json, set_seed


def _norm_path(value: object) -> str:
    return str(value).replace("\\", "/").lower()


def _find_col(df: pd.DataFrame, requested: str | None, candidates: list[str]) -> str | None:
    if requested and requested in df.columns:
        return requested
    lowered = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def load_slice_rows(
    feature_dir: str | Path,
    slice_labels_csv: str | Path,
    split_csv: str | Path,
    split: str,
    case_col: str = "case_id",
    label_col: str = "hemorrhage",
    path_col: str | None = "path",
    instance_col: str | None = "instance_number",
) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
    labels = pd.read_csv(slice_labels_csv)
    case_col = _find_col(labels, case_col, ["case_id", "patient_id", "name", "study_id"])
    label_col = _find_col(labels, label_col, ["hemorrhage", "ich", "label", "target", "is_hemorrhage"])
    path_col = _find_col(labels, path_col, ["path", "dicom_path", "slice_path", "filename", "file"])
    instance_col = _find_col(labels, instance_col, ["instance_number", "slice_index", "slice_idx", "slice"])
    if case_col is None or label_col is None:
        raise ValueError("slice label CSV must contain case and binary label columns")

    labels = labels.copy()
    labels["_case_id"] = labels[case_col].map(normalize_case_id)
    labels["_label"] = pd.to_numeric(labels[label_col], errors="coerce")
    labels = labels.dropna(subset=["_label"])
    labels["_label"] = (labels["_label"] > 0).astype("float32")
    if path_col:
        labels["_path_key"] = labels[path_col].map(_norm_path)
    if instance_col:
        labels["_instance_number"] = pd.to_numeric(labels[instance_col], errors="coerce")

    splits = pd.read_csv(split_csv)
    keep_cases = set(splits.loc[splits["split"] == split, "case_id"].astype(str).map(normalize_case_id))
    labels = labels[labels["_case_id"].isin(keep_cases)]

    feature_dir = Path(feature_dir)
    features: list[torch.Tensor] = []
    targets: list[float] = []
    meta_rows: list[dict[str, object]] = []

    for case_id, case_labels in tqdm(labels.groupby("_case_id"), desc=f"loading {split} slices"):
        feature_path = feature_dir / f"{case_id}.pt"
        if not feature_path.exists():
            continue
        payload = torch.load(feature_path, map_location="cpu")
        case_features = payload["features"].float()
        paths = payload.get("paths", [])
        match_df = pd.DataFrame(
            {
                "_feature_index": np.arange(len(paths)),
                "_feature_path": paths,
                "_path_key": [_norm_path(path) for path in paths],
                "_basename_key": [Path(str(path)).name.lower() for path in paths],
                "_instance_number": [int(Path(str(path)).stem.replace("CT", "") or 0) for path in paths],
            }
        )
        if path_col:
            merged = case_labels.merge(match_df, on="_path_key", how="inner")
            if merged.empty:
                basename_labels = case_labels.copy()
                basename_labels["_basename_key"] = basename_labels[path_col].map(lambda p: Path(str(p)).name.lower())
                merged = basename_labels.merge(match_df, on="_basename_key", how="inner")
        elif instance_col:
            merged = case_labels.merge(match_df, on="_instance_number", how="inner")
        else:
            raise ValueError("slice label CSV must contain path or instance_number/slice_index")
        for _, row in merged.iterrows():
            idx = int(row["_feature_index"])
            features.append(case_features[idx])
            targets.append(float(row["_label"]))
            meta_rows.append(
                {
                    "case_id": case_id,
                    "feature_index": idx,
                    "path": row.get("_feature_path", ""),
                    "label": float(row["_label"]),
                }
            )
    if not features:
        raise ValueError(f"No labeled slices matched cached features for split={split}")
    return torch.stack(features), torch.tensor(targets, dtype=torch.float32), pd.DataFrame(meta_rows)


class SliceHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 0, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_dim and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.net = nn.Sequential(nn.LayerNorm(feature_dim), nn.Dropout(dropout), nn.Linear(feature_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def run_epoch(model, loader, criterion, device, optimizer=None) -> tuple[float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(is_train)
    losses = []
    ys = []
    ps = []
    for x, y in tqdm(loader, leave=False, desc="train" if is_train else "eval"):
        x = x.to(device)
        y = y.to(device)
        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        ys.append(y.detach().cpu().numpy())
        ps.append(torch.sigmoid(logits).detach().cpu().numpy())
    return float(np.mean(losses)), np.concatenate(ys), np.concatenate(ps)


def make_loader(x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vast/slice_head.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))
    device = device_auto()
    paths = cfg["paths"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    out_dir = Path(paths["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_x, train_y, train_meta = load_slice_rows(
        paths["feature_dir"], paths["slice_labels_csv"], paths["split_csv"], "train", **data_cfg
    )
    val_x, val_y, val_meta = load_slice_rows(
        paths["feature_dir"], paths["slice_labels_csv"], paths["split_csv"], "val", **data_cfg
    )
    test_x, test_y, test_meta = load_slice_rows(
        paths["feature_dir"], paths["slice_labels_csv"], paths["split_csv"], "test", **data_cfg
    )

    train_loader = make_loader(train_x, train_y, train_cfg["batch_size"], True, train_cfg["num_workers"])
    val_loader = make_loader(val_x, val_y, train_cfg["batch_size"], False, train_cfg["num_workers"])
    test_loader = make_loader(test_x, test_y, train_cfg["batch_size"], False, train_cfg["num_workers"])

    model = SliceHead(model_cfg["feature_dim"], model_cfg["hidden_dim"], model_cfg["dropout"]).to(device)
    pos_weight = None
    if train_cfg.get("use_pos_weight", True):
        positives = train_y.sum()
        negatives = len(train_y) - positives
        pos_weight = (negatives / positives.clamp_min(1.0)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])

    best_ap = -np.inf
    best_epoch = -1
    wait = 0
    history = []
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        train_loss, train_true, train_prob = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_true, val_prob = run_epoch(model, val_loader, criterion, device)
        val_metrics = binary_metrics(val_true, val_prob)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **{f"val/{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(row)
        score = val_metrics.get("ap", float("nan"))
        if np.isfinite(score) and score > best_ap:
            best_ap = score
            best_epoch = epoch
            wait = 0
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch}, out_dir / "best.pt")
        else:
            wait += 1
        if wait >= int(train_cfg["patience"]):
            break

    checkpoint = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_loss, test_true, test_prob = run_epoch(model, test_loader, criterion, device)
    test_metrics = binary_metrics(test_true, test_prob)
    save_json(
        {"best_epoch": best_epoch, "best_val_ap": best_ap, "test_loss": test_loss, "test": test_metrics},
        out_dir / "metrics.json",
    )
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    pred_df = test_meta.copy()
    pred_df["prob"] = test_prob
    pred_df["pred"] = (test_prob >= 0.5).astype(int)
    pred_df.to_csv(out_dir / "test_slice_predictions.csv", index=False)
    print(test_metrics)


if __name__ == "__main__":
    main()

