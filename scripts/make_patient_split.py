from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def assign_cases(cases: np.ndarray, split_counts: tuple[int, int, int], seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(cases, dtype=str).copy()
    rng.shuffle(shuffled)
    train_n, val_n, test_n = split_counts
    if train_n + val_n + test_n > len(shuffled):
        raise ValueError(f"Requested split counts {split_counts} exceed {len(shuffled)} cases")
    train_cases = shuffled[:train_n]
    val_cases = shuffled[train_n : train_n + val_n]
    test_cases = shuffled[train_n + val_n : train_n + val_n + test_n]
    return {"train": train_cases, "val": val_cases, "test": test_cases}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-csv", default="data/processed/labels.csv")
    parser.add_argument("--out-csv", default="splits/cq500_seed42.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--fixed-cases-csv", default=None)
    parser.add_argument("--fixed-train-count", type=int, default=0)
    parser.add_argument("--fixed-val-count", type=int, default=0)
    parser.add_argument("--fixed-test-count", type=int, default=0)
    args = parser.parse_args()

    labels = pd.read_csv(args.labels_csv)
    cases = labels["case_id"].astype(str).drop_duplicates().to_numpy()
    rows = []
    fixed_assigned: dict[str, np.ndarray] = {}
    fixed_cases = np.array([], dtype=str)
    if args.fixed_cases_csv:
        fixed = pd.read_csv(args.fixed_cases_csv)
        fixed_cases = fixed["case_id"].astype(str).drop_duplicates().to_numpy()
        fixed_cases = np.intersect1d(fixed_cases, cases)
        counts = (args.fixed_train_count, args.fixed_val_count, args.fixed_test_count)
        if any(counts):
            fixed_assigned = assign_cases(fixed_cases, counts, args.seed)
        else:
            raise ValueError("When --fixed-cases-csv is set, pass fixed train/val/test counts")
        for split, split_cases in fixed_assigned.items():
            rows.extend({"case_id": case_id, "split": split, "fixed_group": True} for case_id in np.sort(split_cases))

    remaining_cases = np.setdiff1d(cases, fixed_cases)
    train_cases, test_cases = train_test_split(remaining_cases, test_size=args.test_size, random_state=args.seed)
    rel_val = args.val_size / (1.0 - args.test_size)
    train_cases, val_cases = train_test_split(train_cases, test_size=rel_val, random_state=args.seed)
    for split, split_cases in (("train", train_cases), ("val", val_cases), ("test", test_cases)):
        rows.extend({"case_id": case_id, "split": split, "fixed_group": False} for case_id in np.sort(split_cases))
    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print(out["split"].value_counts().to_string())
    if "fixed_group" in out.columns:
        print(out.groupby(["split", "fixed_group"]).size().to_string())


if __name__ == "__main__":
    main()
