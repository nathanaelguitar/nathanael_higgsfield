#!/usr/bin/env python3
"""Run a text-only Seedance job when provider likeness references are blocked."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from run_fal_seedance import FalSeedanceError, _download, _load_env_file, _result_video_url


DEFAULT_ENDPOINT = "bytedance/seedance-2.5/text-to-video"


def build_arguments(prompt: str, resolution: str, duration: int, aspect_ratio: str) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "resolution": resolution,
        "duration": duration,
        "aspect_ratio": aspect_ratio,
        "generate_audio": True,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model", default=DEFAULT_ENDPOINT)
    parser.add_argument("--resolution", choices=("480p", "720p"), default="720p")
    parser.add_argument("--duration", type=int, default=5, choices=range(4, 16))
    parser.add_argument("--aspect-ratio", default="9:16")
    parser.add_argument("--output", type=Path, default=Path("outputs/fal_seedance_25_text_5s.mp4"))
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path.home() / ".config/ugc_actor_engine/fal.env",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        _load_env_file(args.env_file)
        request = build_arguments(args.prompt, args.resolution, args.duration, args.aspect_ratio)
        if args.dry_run:
            print(json.dumps({"model": args.model, **request}, indent=2))
            return 0
        if not os.environ.get("FAL_KEY", "").strip():
            raise FalSeedanceError("FAL_KEY is not set")
        try:
            import fal_client
        except ImportError as exc:
            raise FalSeedanceError("fal-client is not installed; run ./setup.sh --install-fal") from exc
        print(f"submitting model={args.model}")
        try:
            result = fal_client.subscribe(args.model, arguments=request)
        except Exception as exc:
            raise FalSeedanceError(f"Seedance request failed ({exc.__class__.__name__}): {exc}") from exc
        _download(_result_video_url(result), args.output)
        print(f"output={args.output.resolve()}")
        return 0
    except FalSeedanceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
