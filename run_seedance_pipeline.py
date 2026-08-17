#!/usr/bin/env python3
"""Submit a small, usage-billed Seedance audio/video generation task.

The client deliberately uses only the Python standard library so it can run on
the DGX Spark without installing another CUDA or ML stack.  It accepts local
image/audio files as data URLs for small inputs and supports provider-hosted
URLs/assets when the files are too large or require ModelArk verification.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


BASE_URL = "https://ark.ap-southeast.bytepluses.com/api/v3"
DEFAULT_MODEL = "seedance-1-5-pro-251215"
DEFAULT_PROMPT = (
    "Use Image 1 as the same UGC creator. Keep her identity, hairstyle, "
    "clothing, and vertical selfie framing. She looks directly into the lens "
    "and says exactly: \"That's why you should switch to CanopyChat.\" "
    "Natural eye blinks, subtle head movement, realistic facial micro dynamics, "
    "precise mouth articulation, clean background, no captions, no watermark."
)


class SeedanceError(RuntimeError):
    """Raised for actionable provider or local validation errors."""


def _load_env_file(path: Path) -> None:
    """Load only missing simple KEY=VALUE entries from an ignored env file."""

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


def _api_key(env_file: Path | None) -> str:
    if env_file:
        _load_env_file(env_file)
    key = os.environ.get("ARK_API_KEY", "").strip()
    if not key:
        raise SeedanceError(
            "ARK_API_KEY is not set. Complete BytePlus auth, create an API key, "
            "then export it or use --env-file."
        )
    return key


def _data_url(path: Path) -> str:
    if not path.is_file():
        raise SeedanceError(f"Input file does not exist: {path}")
    mime = mimetypes.guess_type(path.name)[0]
    if mime == "audio/x-wav":
        mime = "audio/wav"
    if mime not in {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff", "audio/wav", "audio/mpeg"}:
        raise SeedanceError(f"Unsupported local input type: {path.suffix or path.name}")
    raw = path.read_bytes()
    # The provider documents 30 MB images, 15 MB audio, and a 64 MB request.
    limit = 30 * 1024 * 1024 if mime.startswith("image/") else 15 * 1024 * 1024
    if len(raw) > limit:
        raise SeedanceError(
            f"{path} is too large for a data URL ({len(raw) / 1024 / 1024:.1f} MiB). "
            "Use --reference-url/--audio-url or a registered asset instead."
        )
    subtype = mime.split("/", 1)[1]
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


def _request(method: str, url: str, api_key: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(detail)
        except json.JSONDecodeError:
            parsed = {"error": {"message": detail[:1000]}}
        message = parsed.get("error", {}).get("message") or parsed.get("message") or detail[:1000]
        raise SeedanceError(f"ModelArk HTTP {exc.code}: {message}") from exc
    except urllib.error.URLError as exc:
        raise SeedanceError(f"ModelArk network error: {exc.reason}") from exc
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise SeedanceError("ModelArk returned a non-JSON response") from exc


def build_request(args: argparse.Namespace) -> dict[str, Any]:
    reference = args.reference_url or (_data_url(args.reference) if args.reference else None)
    audio = args.audio_url or (_data_url(args.audio) if args.audio else None)
    if not reference:
        raise SeedanceError("Provide --reference or --reference-url")
    if not audio:
        raise SeedanceError("Provide --audio or --audio-url")

    content: list[dict[str, Any]] = [
        {"type": "text", "text": args.prompt},
        {
            "type": "image_url",
            "image_url": {"url": reference},
            "role": "reference_image",
        },
        {
            "type": "audio_url",
            "audio_url": {"url": audio},
            "role": "reference_audio",
        },
    ]
    return {
        "model": args.model,
        "content": content,
        "resolution": args.resolution,
        "ratio": args.ratio,
        "duration": args.duration,
        "generate_audio": True,
        "watermark": args.watermark,
    }


def _video_url(result: dict[str, Any]) -> str | None:
    content = result.get("content") or {}
    for key in ("video_url", "file_url", "url"):
        value = content.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, dict) and isinstance(value.get("url"), str):
            return value["url"]
    for key in ("video_url", "file_url"):
        if isinstance(result.get(key), str):
            return result[key]
    return None


def _download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "ugc-actor-engine/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response, output.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        raise SeedanceError(f"Unable to download generated video: {exc}") from exc


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, help="Local portrait image")
    parser.add_argument("--audio", type=Path, help="Local WAV/MP3 reference audio")
    parser.add_argument("--reference-url", help="Public URL or asset:// reference image")
    parser.add_argument("--audio-url", help="Public URL or asset:// reference audio")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--duration", type=int, default=5, choices=range(4, 13))
    parser.add_argument("--resolution", choices=("480p", "720p", "1080p"), default="720p")
    parser.add_argument("--ratio", default="9:16")
    parser.add_argument("--watermark", action="store_true", help="Request a provider watermark")
    parser.add_argument("--output", type=Path, default=Path("outputs/seedance_canopychat_5s.mp4"))
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path.home() / ".config/ugc_actor_engine/byteplus.env",
        help="Ignored file containing ARK_API_KEY=...",
    )
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print request metadata only")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        request_body = build_request(args)
        if args.dry_run:
            print(json.dumps({k: v for k, v in request_body.items() if k != "content"}, indent=2))
            print("content_items=3; media payloads prepared; no API request sent")
            return 0
        api_key = _api_key(args.env_file)
        created = _request("POST", f"{BASE_URL}/contents/generations/tasks", api_key, request_body)
        task_id = created.get("id")
        if not isinstance(task_id, str) or not task_id:
            raise SeedanceError(f"ModelArk did not return a task id: {json.dumps(created)[:1200]}")
        print(f"task_id={task_id}")
        deadline = time.monotonic() + args.timeout_seconds
        while time.monotonic() < deadline:
            result = _request("GET", f"{BASE_URL}/contents/generations/tasks/{task_id}", api_key)
            status = str(result.get("status", "")).lower()
            print(f"status={status or 'unknown'}")
            if status in {"succeeded", "success", "completed"}:
                url = _video_url(result)
                if not url:
                    raise SeedanceError(f"Task succeeded without a video URL: {json.dumps(result)[:1600]}")
                _download(url, args.output)
                print(f"output={args.output.resolve()}")
                return 0
            if status in {"failed", "error", "cancelled", "canceled"}:
                error = result.get("error") or {}
                raise SeedanceError(f"Seedance task {status}: {error.get('message') or json.dumps(result)[:1200]}")
            time.sleep(args.poll_seconds)
        raise SeedanceError(f"Timed out waiting for task {task_id}; it may still be running in ModelArk.")
    except SeedanceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
