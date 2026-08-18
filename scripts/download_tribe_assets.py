#!/usr/bin/env python3
"""Download TRIBE v2 checkpoint dependencies without importing model classes."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


ASSET_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.txt",
    "*.md",
    "*.model",
]

PUBLIC_ENCODERS = (
    ("facebook/vjepa2-vitg-fpc64-256", "facebook_vjepa2-vitg-fpc64-256"),
    ("facebook/w2v-bert-2.0", "facebook_w2v-bert-2.0"),
)
GATED_ENCODERS = (("meta-llama/Llama-3.2-3B", "meta-llama_Llama-3.2-3B"),)


def download(repo_id: str, folder: str, root: Path) -> None:
    destination = root / folder
    print(f"Downloading {repo_id} -> {destination}", flush=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(destination),
        allow_patterns=ASSET_PATTERNS,
        ignore_patterns=["USE_POLICY.md"],
        max_workers=1,
    )
    print(f"Finished {repo_id}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models-root",
        type=Path,
        default=Path("models/tribev2/encoders"),
    )
    parser.add_argument(
        "--include-llama",
        action="store_true",
        help="also download gated meta-llama/Llama-3.2-3B if the HF account has access",
    )
    args = parser.parse_args()
    args.models_root.mkdir(parents=True, exist_ok=True)

    for repo_id, folder in PUBLIC_ENCODERS:
        download(repo_id, folder, args.models_root)
    if args.include_llama:
        for repo_id, folder in GATED_ENCODERS:
            download(repo_id, folder, args.models_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
