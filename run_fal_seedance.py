#!/usr/bin/env python3
"""Run a small, usage-billed Seedance 2.0 job through fal.ai.

This adapter intentionally has no CUDA or local model dependency.  It uploads
the local portrait and audio through fal's authenticated file endpoint, waits
for the hosted generation job, and downloads the returned MP4.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_ENDPOINT = "bytedance/seedance-2.0/fast/reference-to-video"
DEFAULT_PROMPT = (
    "Use @Image1 as the same authorized UGC creator and @Audio1 as the exact "
    "speech and timing reference. Create a photorealistic vertical selfie "
    "marketing video. She looks directly into the camera and says exactly: "
    '\"That\'s why you should switch to CanopyChat.\" '
    "Preserve her identity, hairstyle, clothing, and framing. Use natural eye "
    "blinks, subtle head movement, slight nods, realistic facial micro dynamics, "
    "and precise mouth articulation. No captions, logos, or watermark."
)


class FalSeedanceError(RuntimeError):
    """Raised for actionable local or provider errors."""


def _load_env_file(path: Path) -> None:
    """Load missing simple KEY=VALUE entries from a private env file."""

    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def estimate_cost(duration: int, resolution: str, endpoint: str = DEFAULT_ENDPOINT) -> float | None:
    """Return the published approximate cost for a Seedance 2.0 request.

    Prices are intentionally estimates: fal can change rates and the final
    charge is determined by the provider response/account billing.
    """

    if "fast/reference-to-video" in endpoint:
        rates = {"480p": 0.0, "720p": 0.2419, "1080p": 0.0}
    elif "reference-to-video" in endpoint:
        rates = {"480p": 0.0, "720p": 0.3024, "1080p": 0.682}
    else:
        return None
    rate = rates.get(resolution, 0.0)
    return round(rate * duration, 4) if rate else None


def build_arguments(
    prompt: str,
    image_url: str | None,
    audio_url: str,
    resolution: str,
    duration: int,
    aspect_ratio: str,
    video_url: str | None = None,
    generate_audio: bool = True,
) -> dict[str, Any]:
    """Build the provider payload without importing fal_client."""

    payload: dict[str, Any] = {
        "prompt": prompt,
        "audio_urls": [audio_url],
        "resolution": resolution,
        "duration": duration,
        "aspect_ratio": aspect_ratio,
        "generate_audio": generate_audio,
    }
    if image_url:
        payload["image_urls"] = [image_url]
    if video_url:
        payload["video_urls"] = [video_url]
    return payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    reference = parser.add_mutually_exclusive_group(required=True)
    reference.add_argument("--reference", type=Path, help="Authorized local portrait image")
    reference.add_argument("--reference-video", type=Path, help="Authorized local video reference")
    parser.add_argument("--audio", type=Path, required=True, help="Authorized local WAV/MP3 speech reference")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--model", default=DEFAULT_ENDPOINT, help="fal endpoint/model id")
    parser.add_argument("--resolution", choices=("480p", "720p", "1080p"), default="720p")
    parser.add_argument("--duration", type=int, default=5, choices=range(4, 16))
    parser.add_argument("--aspect-ratio", default="9:16")
    parser.add_argument("--no-generate-audio", action="store_true", help="Keep input audio timing but omit provider-generated audio")
    parser.add_argument("--output", type=Path, default=Path("outputs/fal_seedance_canopychat_5s.mp4"))
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path.home() / ".config/ugc_actor_engine/fal.env",
        help="Private file containing FAL_KEY=... (never commit or share it)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print metadata without submitting")
    return parser.parse_args(argv)


def _validate_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FalSeedanceError(f"{label} does not exist: {path}")
    if path.stat().st_size == 0:
        raise FalSeedanceError(f"{label} is empty: {path}")


def _download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "ugc-actor-engine/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response, output.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        raise FalSeedanceError(f"Unable to download generated video: {exc}") from exc


def _result_video_url(result: dict[str, Any]) -> str:
    video = result.get("video")
    if isinstance(video, dict) and isinstance(video.get("url"), str):
        return video["url"]
    if isinstance(video, str):
        return video
    raise FalSeedanceError(f"fal returned no video URL: {json.dumps(result)[:1600]}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        _load_env_file(args.env_file)
        reference_path = args.reference or args.reference_video
        _validate_file(reference_path, "Reference image/video")
        _validate_file(args.audio, "Audio")
        arguments = build_arguments(
            str(args.prompt),
            str(args.reference) if args.reference else None,
            str(args.audio),
            args.resolution,
            args.duration,
            args.aspect_ratio,
            video_url=str(args.reference_video) if args.reference_video else None,
            generate_audio=not args.no_generate_audio,
        )
        estimate = estimate_cost(args.duration, args.resolution, args.model)
        if args.dry_run:
            metadata = {
                "model": args.model,
                "resolution": args.resolution,
                "duration": args.duration,
                "aspect_ratio": args.aspect_ratio,
                "estimated_cost_usd": estimate,
                "reference": str(args.reference),
                "audio": str(args.audio),
            }
            print(json.dumps(metadata, indent=2))
            print("no fal request sent")
            return 0

        if not os.environ.get("FAL_KEY", "").strip():
            raise FalSeedanceError("FAL_KEY is not set; create a fal key and export it without printing it.")

        try:
            import fal_client
        except ImportError as exc:
            raise FalSeedanceError(
                "fal-client is not installed; run ./setup.sh --install-fal"
            ) from exc

        try:
            print(f"uploading reference={reference_path}")
            reference_url = fal_client.upload_file(str(reference_path))
            print(f"uploading audio={args.audio}")
            audio_url = fal_client.upload_file(str(args.audio))
            arguments = build_arguments(
                args.prompt,
                reference_url if args.reference else None,
                audio_url,
                args.resolution,
                args.duration,
                args.aspect_ratio,
                video_url=reference_url if args.reference_video else None,
                generate_audio=not args.no_generate_audio,
            )
            if estimate is not None:
                print(f"estimated_cost_usd={estimate:.4f}")
            print(f"submitting model={args.model}")
            result = fal_client.subscribe(args.model, arguments=arguments)
        except Exception as exc:
            raise FalSeedanceError(
                f"fal request failed ({exc.__class__.__name__}): {exc}"
            ) from exc
        output_url = _result_video_url(result)
        _download(output_url, args.output)
        print(f"output={args.output.resolve()}")
        return 0
    except FalSeedanceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
