from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from ctmil.env import get_hf_token, load_project_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--local-dir", default="data/raw/hf/dinov3-vitb16-pretrain-lvd1689m")
    args = parser.parse_args()

    load_project_env()
    token = get_hf_token()
    Path(args.local_dir).mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        token=token,
        local_dir_use_symlinks=False,
    )
    print(path)


if __name__ == "__main__":
    main()

