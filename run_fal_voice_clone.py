#!/usr/bin/env python3
"""Generate an authorized reference-voice speech clip through fal F5-TTS."""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from run_fal_seedance import FalSeedanceError, _download, _load_env_file


ENDPOINT = "fal-ai/f5-tts"
DEFAULT_TEXT = "That's why you should switch to CanopyChat."


def build_arguments(reference_url: str, text: str, reference_text: str | None, model_type: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "gen_text": text,
        "ref_audio_url": reference_url,
        "model_type": model_type,
        "remove_silence": True,
    }
    if reference_text:
        payload["ref_text"] = reference_text
    return payload


def _audio_url(result: dict[str, Any]) -> str:
    value = result.get("audio_url")
    if isinstance(value, dict) and isinstance(value.get("url"), str):
        return value["url"]
    if isinstance(value, str):
        return value
    raise FalSeedanceError(f"F5-TTS returned no audio URL: {str(result)[:1200]}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-audio", type=Path, required=True)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--reference-text", help="Optional transcript; F5-TTS can ASR it when omitted")
    parser.add_argument("--model-type", choices=("F5-TTS", "E2-TTS"), default="F5-TTS")
    parser.add_argument("--output", type=Path, default=Path("outputs/canopychat_inputs/fal_voice_clone_canopychat.wav"))
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path.home() / ".config/ugc_actor_engine/fal.env",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        _load_env_file(args.env_file)
        if not args.reference_audio.is_file():
            raise FalSeedanceError(f"Reference audio does not exist: {args.reference_audio}")
        if not os.environ.get("FAL_KEY", "").strip():
            raise FalSeedanceError("FAL_KEY is not set")
        try:
            import fal_client
        except ImportError as exc:
            raise FalSeedanceError("fal-client is not installed; run ./setup.sh --install-fal") from exc

        print(f"uploading voice reference={args.reference_audio}")
        reference_url = fal_client.upload_file(str(args.reference_audio))
        arguments = build_arguments(reference_url, args.text, args.reference_text, args.model_type)
        print(f"submitting model={ENDPOINT}")
        try:
            result = fal_client.subscribe(ENDPOINT, arguments=arguments)
        except Exception as exc:
            raise FalSeedanceError(f"F5-TTS request failed ({exc.__class__.__name__}): {exc}") from exc
        _download(_audio_url(result), args.output)
        print(f"output={args.output.resolve()}")
        return 0
    except FalSeedanceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
