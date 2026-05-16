from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-csv", default="data/processed/labels.csv")
    parser.add_argument("--out-csv", default="splits/cq500_seed42.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    args = parser.parse_args()

    labels = pd.read_csv(args.labels_csv)
    cases = labels["case_id"].astype(str).drop_duplicates().to_numpy()
    train_cases, test_cases = train_test_split(cases, test_size=args.test_size, random_state=args.seed)
    rel_val = args.val_size / (1.0 - args.test_size)
    train_cases, val_cases = train_test_split(train_cases, test_size=rel_val, random_state=args.seed)
    rows = []
    for split, split_cases in (("train", train_cases), ("val", val_cases), ("test", test_cases)):
        rows.extend({"case_id": case_id, "split": split} for case_id in np.sort(split_cases))
    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print(out["split"].value_counts().to_string())


if __name__ == "__main__":
    main()

