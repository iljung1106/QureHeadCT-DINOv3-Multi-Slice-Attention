from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def load_project_env(root: str | Path = ".") -> None:
    root = Path(root)
    for name in (".env", ".env.txt"):
        path = root / name
        if path.exists():
            load_dotenv(path, override=False)


def get_hf_token() -> str | None:
    return os.getenv("HF_KEY") or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

