from __future__ import annotations

import argparse

from ctmil.labels import normalize_label_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--out-csv", default="data/processed/labels.csv")
    parser.add_argument("--id-column", default=None)
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["ICH", "IPH", "IVH", "SDH", "EDH", "SAH", "MidlineShift", "MassEffect"],
    )
    args = parser.parse_args()
    normalize_label_csv(args.labels_csv, args.out_csv, args.labels, args.id_column)
    print(args.out_csv)


if __name__ == "__main__":
    main()

