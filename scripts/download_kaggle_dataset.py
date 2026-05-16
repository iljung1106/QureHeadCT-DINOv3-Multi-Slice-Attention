from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZipFile

from ctmil.env import load_project_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="crawford/qureai-headct")
    parser.add_argument("--out-dir", default="data/raw/qureai-headct")
    parser.add_argument("--unzip", action="store_true")
    args = parser.parse_args()

    load_project_env()
    from kaggle.api.kaggle_api_extended import KaggleApi

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(args.dataset, path=str(out_dir), unzip=False)
    if args.unzip:
        for zip_path in out_dir.glob("*.zip"):
            with ZipFile(zip_path) as zf:
                zf.extractall(out_dir)
    print(out_dir)


if __name__ == "__main__":
    main()
