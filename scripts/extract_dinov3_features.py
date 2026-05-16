from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from ctmil.dicom_utils import ct_to_rgb_windows, load_hu
from ctmil.env import get_hf_token, load_project_env
from ctmil.train_utils import device_auto


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-csv", default="data/processed/dicom_index.csv")
    parser.add_argument("--split-csv", default=None)
    parser.add_argument("--split", default=None, choices=[None, "train", "val", "test"])
    parser.add_argument("--model", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--out-dir", default="data/features/dinov3_vitb16")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-slices", type=int, default=0)
    parser.add_argument("--limit-cases", type=int, default=0)
    args = parser.parse_args()

    load_project_env()
    token = get_hf_token()
    device = device_auto()
    processor = AutoImageProcessor.from_pretrained(args.model, token=token)
    model = AutoModel.from_pretrained(args.model, token=token).to(device).eval()

    index = pd.read_csv(args.index_csv)
    if args.split_csv and args.split:
        split_df = pd.read_csv(args.split_csv)
        keep = set(split_df.loc[split_df["split"] == args.split, "case_id"].astype(str))
        index = index[index["case_id"].astype(str).isin(keep)]
    if args.limit_cases:
        case_ids = sorted(index["case_id"].astype(str).unique())[: args.limit_cases]
        index = index[index["case_id"].astype(str).isin(case_ids)]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sort_cols = ["study_uid", "series_uid", "image_position_z", "instance_number"]
    for case_id, group in tqdm(index.groupby(index["case_id"].astype(str)), desc="Cases"):
        out_path = out_dir / f"{case_id}.pt"
        if out_path.exists():
            continue
        group = group.sort_values(sort_cols)
        if args.max_slices and len(group) > args.max_slices:
            idx = torch.linspace(0, len(group) - 1, args.max_slices).long().tolist()
            group = group.iloc[idx]
        embeddings = []
        paths = group["path"].tolist()
        for start in range(0, len(paths), args.batch_size):
            images = []
            batch_paths = paths[start : start + args.batch_size]
            for path in batch_paths:
                hu = load_hu(path)
                rgb = ct_to_rgb_windows(hu)
                images.append(Image.fromarray(rgb))
            inputs = processor(images=images, return_tensors="pt").to(device)
            outputs = model(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                feat = outputs.pooler_output
            else:
                feat = outputs.last_hidden_state[:, 0]
            embeddings.append(feat.detach().cpu())
        features = torch.cat(embeddings, dim=0)
        torch.save({"case_id": case_id, "features": features, "paths": paths}, out_path)


if __name__ == "__main__":
    main()
