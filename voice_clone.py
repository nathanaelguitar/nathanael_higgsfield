#!/usr/bin/env python3
"""Offline reference-audio voice cloning adapter for NeuTTS."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthesize speech in an authorized reference voice")
    parser.add_argument("--reference-audio", type=Path, required=True)
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone", default="neuphonic/neutts-air")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if not args.reference_audio.is_file():
        raise SystemExit(f"Reference audio does not exist: {args.reference_audio}")
    if not args.reference_text.strip():
        raise SystemExit("Reference transcript cannot be empty")

    try:
        import soundfile as sf
        import torch
        from neutts import NeuTTS
    except ImportError as exc:
        raise SystemExit(
            "Voice cloning requires the optional .voice_venv; run setup.sh --install-voice-clone."
        ) from exc

    # NeuCodec expects 24 kHz mono input. Normalize here so callers can pass
    # the original camera/video track without knowing that detail.
    # PyTorch's ARM64 oneDNN depthwise-convolution path currently raises an
    # Xbyak immediate-parameter error for NeuCodec. The native kernel is stable
    # on DGX Spark and still keeps this small speech stage CPU-only.
    torch.backends.mkldnn.enabled = False
    torch.set_num_threads(min(torch.get_num_threads(), 4))

    with tempfile.TemporaryDirectory(prefix="neutts-ref-") as temp_dir:
        normalized = Path(temp_dir) / "reference_24k.wav"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", str(args.reference_audio),
                 "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(normalized)],
                check=True,
            )
        except FileNotFoundError as exc:
            raise SystemExit("Voice cloning requires ffmpeg on PATH.") from exc
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"Could not normalize reference audio (ffmpeg exit {exc.returncode}).") from exc

        tts = NeuTTS(backbone_repo=args.backbone, backbone_device=args.device, codec_device=args.device)
        reference_codes = tts.encode_reference(str(normalized))
        samples = tts.infer(args.text, reference_codes, args.reference_text.strip())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output), samples, 24_000)
    print(f"[voice] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
