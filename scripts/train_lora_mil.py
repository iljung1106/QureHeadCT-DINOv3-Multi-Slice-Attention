from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from torch import nn
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from ctmil.dicom_utils import ct_to_rgb_windows, load_hu
from ctmil.env import get_hf_token, load_project_env
from ctmil.labels import normalize_case_id
from ctmil.metrics import binary_metrics, multilabel_metrics
from ctmil.models import DINOv3BiGRUGatedABMIL
from ctmil.train_utils import device_auto, load_yaml, save_json, set_seed


def norm_path(value: object) -> str:
    return str(value).replace("\\", "/").lower()


def load_cases(index_csv: str, labels_csv: str, split_csv: str, split: str, label_cols: list[str]) -> list[dict]:
    index = pd.read_csv(index_csv)
    index["case_id"] = index["case_id"].map(normalize_case_id)
    labels = pd.read_csv(labels_csv)
    labels["case_id"] = labels["case_id"].map(normalize_case_id)
    splits = pd.read_csv(split_csv)
    splits["case_id"] = splits["case_id"].map(normalize_case_id)
    split_cases = set(splits.loc[splits["split"] == split, "case_id"])
    labels = labels[labels["case_id"].isin(split_cases)]
    rows = []
    sort_cols = ["study_uid", "series_uid", "image_position_z", "instance_number"]
    for _, label_row in labels.iterrows():
        case_id = label_row["case_id"]
        group = index[index["case_id"] == case_id].sort_values(sort_cols)
        if group.empty:
            continue
        rows.append(
            {
                "case_id": case_id,
                "paths": group["path"].tolist(),
                "scan_labels": label_row[label_cols].astype(float).to_numpy(dtype=np.float32),
            }
        )
    return rows


def load_slice_label_map(slice_labels_csv: str | None) -> dict[tuple[str, str], float]:
    if not slice_labels_csv or not Path(slice_labels_csv).exists():
        return {}
    df = pd.read_csv(slice_labels_csv)
    required = {"case_id", "path", "hemorrhage"}
    if not required.issubset(df.columns):
        raise ValueError(f"{slice_labels_csv} must contain columns: {sorted(required)}")
    out: dict[tuple[str, str], float] = {}
    for _, row in df.iterrows():
        out[(normalize_case_id(row["case_id"]), norm_path(row["path"]))] = float(row["hemorrhage"] > 0)
    return out


def subsample_paths(paths: list[str], max_slices: int, train: bool) -> list[str]:
    if max_slices <= 0 or len(paths) <= max_slices:
        return paths
    if train:
        idx = np.sort(np.random.choice(len(paths), size=max_slices, replace=False))
    else:
        idx = np.linspace(0, len(paths) - 1, max_slices).round().astype(int)
    return [paths[i] for i in idx]


def load_images(paths: list[str]) -> list[Image.Image]:
    images = []
    for path in paths:
        hu = load_hu(path)
        rgb = ct_to_rgb_windows(hu)
        images.append(Image.fromarray(rgb))
    return images


def infer_lora_targets(model: nn.Module, requested: list[str]) -> list[str]:
    requested = list(dict.fromkeys(requested))
    linear_leaf_names = {
        name.split(".")[-1] for name, module in model.named_modules() if isinstance(module, nn.Linear)
    }
    selected = [name for name in requested if name in linear_leaf_names]
    if selected:
        return selected
    fallback = [name for name in linear_leaf_names if any(key in name.lower() for key in ("q", "k", "v"))]
    if fallback:
        return sorted(fallback)
    raise ValueError(f"Could not infer LoRA target modules from requested={requested}")


def pooled_features(outputs) -> torch.Tensor:
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output
    return outputs.last_hidden_state[:, 0]


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


class LoraMILModel(nn.Module):
    def __init__(self, encoder: nn.Module, mil_head: nn.Module, slice_head: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.mil_head = mil_head
        self.slice_head = slice_head


def encode_slices(
    model: LoraMILModel,
    processor,
    paths: list[str],
    device: torch.device,
    slice_batch_size: int,
    amp: bool,
) -> torch.Tensor:
    chunks = []
    use_amp = amp and device.type == "cuda"
    for start in range(0, len(paths), slice_batch_size):
        batch_paths = paths[start : start + slice_batch_size]
        images = load_images(batch_paths)
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model.encoder(**inputs)
            features = pooled_features(outputs)
        chunks.append(features.float())
    return torch.cat(chunks, dim=0)


def case_forward(
    model: LoraMILModel,
    processor,
    case: dict,
    slice_label_map: dict[tuple[str, str], float],
    device: torch.device,
    max_slices: int,
    slice_batch_size: int,
    amp: bool,
    train: bool,
) -> dict[str, torch.Tensor]:
    case_id = case["case_id"]
    paths = subsample_paths(case["paths"], max_slices=max_slices, train=train)
    features = encode_slices(model, processor, paths, device, slice_batch_size, amp)
    scan_labels = torch.tensor(case["scan_labels"], dtype=torch.float32, device=device).unsqueeze(0)
    mask = torch.ones(1, features.shape[0], dtype=torch.bool, device=device)
    lengths = torch.tensor([features.shape[0]], dtype=torch.long, device=device)
    mil_out = model.mil_head(features.unsqueeze(0), mask, lengths)
    slice_logits = model.slice_head(features)
    slice_targets = []
    slice_mask = []
    for path in paths:
        key = (case_id, norm_path(path))
        if key in slice_label_map:
            slice_targets.append(slice_label_map[key])
            slice_mask.append(True)
        else:
            slice_targets.append(0.0)
            slice_mask.append(False)
    return {
        "scan_logits": mil_out["logits"],
        "scan_labels": scan_labels,
        "slice_logits": slice_logits,
        "slice_targets": torch.tensor(slice_targets, dtype=torch.float32, device=device),
        "slice_mask": torch.tensor(slice_mask, dtype=torch.bool, device=device),
    }


def scan_pos_weight(labels_csv: str, split_csv: str, labels: list[str], device: torch.device) -> torch.Tensor:
    df = pd.read_csv(labels_csv).merge(pd.read_csv(split_csv), on="case_id", how="inner")
    train = df[df["split"] == "train"][labels].astype(float)
    positives = torch.tensor(train.sum(axis=0).to_numpy(), dtype=torch.float32)
    negatives = torch.tensor(len(train) - train.sum(axis=0).to_numpy(), dtype=torch.float32)
    return (negatives / positives.clamp_min(1.0)).to(device)


def slice_pos_weight(slice_label_map: dict[tuple[str, str], float], train_cases: list[dict], device: torch.device) -> torch.Tensor | None:
    values = []
    train_case_ids = {case["case_id"] for case in train_cases}
    for (case_id, _), value in slice_label_map.items():
        if case_id in train_case_ids:
            values.append(value)
    if not values or sum(values) == 0:
        return None
    positives = torch.tensor(sum(values), dtype=torch.float32)
    negatives = torch.tensor(len(values) - sum(values), dtype=torch.float32)
    return (negatives / positives.clamp_min(1.0)).to(device)


def evaluate(
    model: LoraMILModel,
    processor,
    cases: list[dict],
    slice_label_map: dict[tuple[str, str], float],
    scan_criterion,
    slice_criterion,
    cfg: dict,
    device: torch.device,
) -> dict:
    model.eval()
    scan_losses = []
    slice_losses = []
    scan_true = []
    scan_prob = []
    slice_true = []
    slice_prob = []
    train_cfg = cfg["training"]
    with torch.no_grad():
        for case in tqdm(cases, desc="eval", leave=False):
            out = case_forward(
                model,
                processor,
                case,
                slice_label_map,
                device,
                train_cfg["max_slices"],
                train_cfg["slice_batch_size"],
                train_cfg.get("amp", True),
                train=False,
            )
            scan_loss = scan_criterion(out["scan_logits"], out["scan_labels"])
            scan_losses.append(float(scan_loss.detach().cpu()))
            scan_true.append(out["scan_labels"].detach().cpu().numpy())
            scan_prob.append(torch.sigmoid(out["scan_logits"]).detach().cpu().numpy())
            if out["slice_mask"].any():
                logits = out["slice_logits"][out["slice_mask"]]
                targets = out["slice_targets"][out["slice_mask"]]
                loss = slice_criterion(logits, targets)
                slice_losses.append(float(loss.detach().cpu()))
                slice_true.append(targets.detach().cpu().numpy())
                slice_prob.append(torch.sigmoid(logits).detach().cpu().numpy())
    metrics = {
        "scan_loss": float(np.mean(scan_losses)) if scan_losses else float("nan"),
        "slice_loss": float(np.mean(slice_losses)) if slice_losses else float("nan"),
    }
    metrics.update({f"scan/{k}": v for k, v in multilabel_metrics(np.concatenate(scan_true), np.concatenate(scan_prob), cfg["labels"]).items()})
    if slice_true:
        metrics.update({f"slice/{k}": v for k, v in binary_metrics(np.concatenate(slice_true), np.concatenate(slice_prob)).items()})
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vast/lora_mil.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))
    load_project_env()
    device = device_auto()
    paths_cfg = cfg["paths"]
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    labels = cfg["labels"]
    out_dir = Path(paths_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_cases = load_cases(paths_cfg["index_csv"], paths_cfg["labels_csv"], paths_cfg["split_csv"], "train", labels)
    val_cases = load_cases(paths_cfg["index_csv"], paths_cfg["labels_csv"], paths_cfg["split_csv"], "val", labels)
    test_cases = load_cases(paths_cfg["index_csv"], paths_cfg["labels_csv"], paths_cfg["split_csv"], "test", labels)
    slice_labels = load_slice_label_map(paths_cfg.get("slice_labels_csv"))

    token = get_hf_token()
    processor = AutoImageProcessor.from_pretrained(paths_cfg["model_name_or_path"], token=token)
    encoder = AutoModel.from_pretrained(paths_cfg["model_name_or_path"], token=token)
    target_modules = infer_lora_targets(encoder, cfg["lora"]["target_modules"])
    print("LoRA target modules:", target_modules)
    lora_cfg = LoraConfig(
        r=int(cfg["lora"]["r"]),
        lora_alpha=int(cfg["lora"]["alpha"]),
        lora_dropout=float(cfg["lora"]["dropout"]),
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    encoder = get_peft_model(encoder, lora_cfg)
    encoder.print_trainable_parameters()

    mil_head = DINOv3BiGRUGatedABMIL(
        feature_dim=model_cfg["feature_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_labels=len(labels),
        gru_layers=model_cfg["gru_layers"],
        attention_dim=model_cfg["attention_dim"],
        dropout=model_cfg["dropout"],
    )
    slice_head = SliceHead(model_cfg["feature_dim"], model_cfg.get("slice_head_hidden_dim", 0), model_cfg["dropout"])
    model = LoraMILModel(encoder, mil_head, slice_head).to(device)

    scan_pw = scan_pos_weight(paths_cfg["labels_csv"], paths_cfg["split_csv"], labels, device) if train_cfg.get("use_scan_pos_weight", True) else None
    slice_pw = slice_pos_weight(slice_labels, train_cases, device) if train_cfg.get("use_slice_pos_weight", True) else None
    scan_criterion = nn.BCEWithLogitsLoss(pos_weight=scan_pw)
    slice_criterion = nn.BCEWithLogitsLoss(pos_weight=slice_pw)

    lora_params = [p for n, p in model.encoder.named_parameters() if p.requires_grad]
    head_params = list(model.mil_head.parameters()) + list(model.slice_head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": float(train_cfg["lr_lora"])},
            {"params": head_params, "lr": float(train_cfg["lr_head"])},
        ],
        weight_decay=float(train_cfg["weight_decay"]),
    )

    best_auc = -np.inf
    best_epoch = -1
    best_state = None
    wait = 0
    history = []
    accum = int(train_cfg.get("grad_accum_steps", 1))
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        np.random.shuffle(train_cases)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, case in enumerate(tqdm(train_cases, desc=f"epoch {epoch}"), start=1):
            out = case_forward(
                model,
                processor,
                case,
                slice_labels,
                device,
                train_cfg["max_slices"],
                train_cfg["slice_batch_size"],
                train_cfg.get("amp", True),
                train=True,
            )
            scan_loss = scan_criterion(out["scan_logits"], out["scan_labels"])
            loss = float(train_cfg["scan_loss_weight"]) * scan_loss
            if out["slice_mask"].any() and float(train_cfg["slice_loss_weight"]) > 0:
                slice_loss = slice_criterion(out["slice_logits"][out["slice_mask"]], out["slice_targets"][out["slice_mask"]])
                loss = loss + float(train_cfg["slice_loss_weight"]) * slice_loss
            (loss / accum).backward()
            if step % accum == 0 or step == len(train_cases):
                if train_cfg.get("grad_clip", 0):
                    nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate(model, processor, val_cases, slice_labels, scan_criterion, slice_criterion, cfg, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val/{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(row)
        score = val_metrics.get("scan/macro_auc", float("nan"))
        if np.isfinite(score) and score > best_auc:
            best_auc = score
            best_epoch = epoch
            wait = 0
            model.encoder.save_pretrained(out_dir / "best_lora")
            torch.save(
                {
                    "mil_head": model.mil_head.state_dict(),
                    "slice_head": model.slice_head.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                },
                out_dir / "best_heads.pt",
            )
            best_state = {
                "lora": {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model.encoder).items()},
                "mil_head": {k: v.detach().cpu().clone() for k, v in model.mil_head.state_dict().items()},
                "slice_head": {k: v.detach().cpu().clone() for k, v in model.slice_head.state_dict().items()},
            }
        else:
            wait += 1
        if wait >= int(train_cfg["patience"]):
            break

    if best_state is not None:
        set_peft_model_state_dict(model.encoder, {k: v.to(device) for k, v in best_state["lora"].items()})
        model.mil_head.load_state_dict(best_state["mil_head"])
        model.slice_head.load_state_dict(best_state["slice_head"])
    test_metrics = evaluate(model, processor, test_cases, slice_labels, scan_criterion, slice_criterion, cfg, device)
    save_json({"best_epoch": best_epoch, "best_val_macro_auc": best_auc, "test": test_metrics}, out_dir / "metrics.json")
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    print(test_metrics)


if __name__ == "__main__":
    main()
