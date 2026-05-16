from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ctmil.data import FeatureBagDataset, collate_feature_bags
from ctmil.metrics import multilabel_metrics
from ctmil.models import DINOv3BiGRUGatedABMIL
from ctmil.train_utils import device_auto, load_yaml, save_json, set_seed


def run_epoch(model, loader, criterion, device, optimizer=None, grad_clip=1.0):
    is_train = optimizer is not None
    model.train(is_train)
    losses = []
    ys = []
    ps = []
    case_ids = []
    for batch in tqdm(loader, leave=False, desc="train" if is_train else "eval"):
        features = batch["features"].to(device)
        mask = batch["mask"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)
        with torch.set_grad_enabled(is_train):
            outputs = model(features, mask, lengths)
            loss = criterion(outputs["logits"], labels)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        ys.append(labels.detach().cpu().numpy())
        ps.append(torch.sigmoid(outputs["logits"]).detach().cpu().numpy())
        case_ids.extend(batch["case_id"])
    return {
        "loss": float(np.mean(losses)),
        "y": np.concatenate(ys, axis=0),
        "p": np.concatenate(ps, axis=0),
        "case_id": case_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    labels = cfg["labels"]
    set_seed(int(cfg["seed"]))
    device = device_auto()
    paths = cfg["paths"]
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    out_dir = Path(paths["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = FeatureBagDataset(paths["labels_csv"], paths["split_csv"], paths["feature_dir"], "train", labels)
    val_ds = FeatureBagDataset(paths["labels_csv"], paths["split_csv"], paths["feature_dir"], "val", labels)
    test_ds = FeatureBagDataset(paths["labels_csv"], paths["split_csv"], paths["feature_dir"], "test", labels)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_feature_bags,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_feature_bags,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_feature_bags,
    )

    model = DINOv3BiGRUGatedABMIL(
        feature_dim=model_cfg["feature_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_labels=len(labels),
        gru_layers=model_cfg["gru_layers"],
        attention_dim=model_cfg["attention_dim"],
        dropout=model_cfg["dropout"],
    ).to(device)

    pos_weight = None
    if train_cfg.get("use_pos_weight", True):
        train_labels = pd.read_csv(paths["labels_csv"]).merge(
            pd.read_csv(paths["split_csv"]), on="case_id", how="inner"
        )
        train_labels = train_labels[train_labels["split"] == "train"][labels].astype(float)
        positives = torch.tensor(train_labels.sum(axis=0).to_numpy(), dtype=torch.float32)
        negatives = torch.tensor(len(train_labels) - train_labels.sum(axis=0).to_numpy(), dtype=torch.float32)
        pos_weight = (negatives / positives.clamp_min(1.0)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    best_auc = -np.inf
    best_epoch = -1
    wait = 0
    history = []
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        train_result = run_epoch(
            model, train_loader, criterion, device, optimizer, float(train_cfg["grad_clip"])
        )
        val_result = run_epoch(model, val_loader, criterion, device)
        val_metrics = multilabel_metrics(val_result["y"], val_result["p"], labels)
        row = {
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "val_loss": val_result["loss"],
            **val_metrics,
        }
        history.append(row)
        print(row)
        score = val_metrics.get("macro_auc", float("nan"))
        if np.isfinite(score) and score > best_auc:
            best_auc = score
            best_epoch = epoch
            wait = 0
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch}, out_dir / "best.pt")
        else:
            wait += 1
        if wait >= int(train_cfg["patience"]):
            break

    checkpoint = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_result = run_epoch(model, test_loader, criterion, device)
    test_metrics = multilabel_metrics(test_result["y"], test_result["p"], labels)
    save_json({"best_epoch": best_epoch, "best_val_macro_auc": best_auc, "test": test_metrics}, out_dir / "metrics.json")
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    pred_df = pd.DataFrame(test_result["p"], columns=[f"{label}_prob" for label in labels])
    pred_df.insert(0, "case_id", test_result["case_id"])
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)
    print(test_metrics)


if __name__ == "__main__":
    main()

