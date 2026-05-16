from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path
from zipfile import ZipFile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default="https://zenodo.org/records/8063221/files/Seg-CQ500.zip?download=1",
    )
    parser.add_argument("--out-dir", default="data/raw/seg-cq500")
    parser.add_argument("--unzip", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "Seg-CQ500.zip"
    if not zip_path.exists():
        print(f"Downloading {args.url} -> {zip_path}")
        urllib.request.urlretrieve(args.url, zip_path)
    else:
        print(f"Using existing {zip_path}")
    if args.unzip:
        with ZipFile(zip_path) as zf:
            zf.extractall(out_dir)
    print(out_dir)


if __name__ == "__main__":
    main()

