#!/usr/bin/env python3
"""Modular local UGC actor pipeline.

Heavy model runtimes stay behind subprocess adapters so orchestration and
FFmpeg validation work before large checkpoints are installed.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS = PROJECT_ROOT / "models"
DEFAULT_OUTPUTS = PROJECT_ROOT / "outputs"


class PipelineError(RuntimeError):
    """An actionable pipeline failure."""


@dataclass
class PipelineConfig:
    reference: Path | None = None
    prompt: str | None = None
    audio: Path | None = None
    output: Path = DEFAULT_OUTPUTS / "ugc_actor.mp4"
    duration: float | None = None
    width: int = 1080
    height: int = 1920
    fps: int = 60
    foundation_backend: str = "auto"
    animation_backend: str = "auto"
    enhancer: str = "auto"
    models_root: Path = DEFAULT_MODELS
    repos_root: Path = PROJECT_ROOT / "third_party"
    work_root: Path | None = None
    foundation_command: str | None = None
    animation_command: str | None = None
    enhancer_command: str | None = None
    echo_sample_size: int = 512
    echo_steps: int = 8
    echo_memory_mode: str = "sequential_cpu_offload"
    allow_fallback: bool = False
    demo: bool = False
    keep_intermediates: bool = False
    portrait_prompt: str = "A natural, confident UGC creator speaking directly to camera"
    seed: int = 43
    notes: list[str] = field(default_factory=list)


@dataclass
class StageArtifact:
    name: str
    path: Path
    backend: str
    duration: float
    metadata: dict[str, Any] = field(default_factory=dict)


def _command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def _run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("[run]", " ".join(shlex.quote(x) for x in command), flush=True)
    try:
        subprocess.run(command, cwd=cwd, env=env, check=True)
    except FileNotFoundError as exc:
        raise PipelineError(f"Executable not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise PipelineError(f"Command failed with exit code {exc.returncode}: {command[0]}") from exc


def _require_ffmpeg() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if not _command_exists(tool)]
    if missing:
        raise PipelineError("Missing required system tools: " + ", ".join(missing))


def _probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    try:
        return max(0.01, float(result.stdout.strip()))
    except ValueError as exc:
        raise PipelineError(f"Could not read media duration from {path}") from exc


def _probe_fps(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
         "-of", "default=nw=1:nk=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    value = result.stdout.strip()
    try:
        numerator, denominator = value.split("/", 1)
        fps = float(numerator) / float(denominator)
        if fps <= 0:
            raise ValueError
        return fps
    except (ValueError, ZeroDivisionError) as exc:
        raise PipelineError(f"Could not read video FPS from {path}") from exc


def _validate_input(config: PipelineConfig) -> float:
    _require_ffmpeg()
    if config.audio is None:
        raise PipelineError("An input voice audio file is required (use --audio or --demo).")
    if not config.audio.is_file():
        raise PipelineError(f"Audio file does not exist: {config.audio}")
    if config.reference is None and not config.prompt:
        raise PipelineError("Provide --reference or --prompt.")
    if config.reference is not None and not config.reference.is_file():
        raise PipelineError(f"Reference image does not exist: {config.reference}")
    duration = config.duration or _probe_duration(config.audio)
    if duration <= 0 or duration > 3600:
        raise PipelineError("Duration must be greater than zero and no more than one hour.")
    if config.width % 2 or config.height % 2 or config.width < 256 or config.height < 256:
        raise PipelineError("Output dimensions must be even and at least 256 pixels.")
    if config.fps < 1 or config.fps > 120:
        raise PipelineError("FPS must be between 1 and 120.")
    return duration


def _configure_runtime() -> dict[str, str]:
    """Set conservative allocator defaults for unified memory and stage isolation."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", os.environ["PYTORCH_CUDA_ALLOC_CONF"])
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    env = dict(os.environ)
    compat = PROJECT_ROOT / "compat"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(compat), env.get("PYTHONPATH", ""))))
    return env


def _release_torch_memory() -> None:
    gc.collect()
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _render_command(template: str, values: dict[str, Any]) -> list[str]:
    safe_values = {key: shlex.quote(str(value)) for key, value in values.items()}
    try:
        return shlex.split(template.format(**safe_values))
    except KeyError as exc:
        raise PipelineError(f"Command template references unknown placeholder: {exc.args[0]}") from exc


def _copy_file(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def _valid_safetensors(path: Path) -> bool:
    """Reject interrupted/mislabeled downloads before model loading."""
    try:
        header_size = int.from_bytes(path.read_bytes()[:8], "little")
        return 0 < header_size < 10_000_000 and header_size + 8 < path.stat().st_size
    except (OSError, ValueError):
        return False


def _make_portrait_card(path: Path, *, width: int = 768, height: int = 1024) -> None:
    """Create a neutral synthetic demo portrait; never used for production input."""
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise PipelineError("Pillow is needed only for --demo; run setup.sh first.") from exc
    image = Image.new("RGB", (width, height), (22, 28, 42))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, int(height * 0.68), width, height), fill=(35, 58, 79))
    cx, cy = width // 2, int(height * 0.37)
    draw.ellipse((cx - 155, cy - 210, cx + 155, cy + 150), fill=(191, 139, 105), outline=(245, 202, 159), width=5)
    draw.pieslice((cx - 165, cy - 235, cx + 165, cy + 80), 180, 360, fill=(38, 28, 28))
    draw.ellipse((cx - 88, cy - 60, cx - 54, cy - 26), fill=(18, 20, 25))
    draw.ellipse((cx + 54, cy - 60, cx + 88, cy - 26), fill=(18, 20, 25))
    draw.arc((cx - 70, cy + 20, cx + 70, cy + 105), 10, 170, fill=(85, 35, 38), width=8)
    draw.rounded_rectangle((cx - 205, int(height * 0.58), cx + 205, height + 80), radius=75, fill=(69, 102, 135))
    image.save(path)


def _make_demo_audio(path: Path, duration: float = 10.0) -> None:
    _run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=48000", "-t", str(duration), "-af", "volume=0.12", "-c:a", "pcm_s16le", str(path)])


def _make_image_video(image: Path, output: Path, *, duration: float, width: int, height: int, fps: int) -> None:
    vf = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},format=yuv420p"
    _run([
        "ffmpeg", "-y", "-loop", "1", "-framerate", str(fps), "-i", str(image), "-t", f"{duration:.3f}",
        "-vf", vf, "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(output),
    ])


def _stage_one(config: PipelineConfig, duration: float, work: Path, env: dict[str, str]) -> StageArtifact:
    output = work / "stage1_foundation.mp4"
    backend = config.foundation_backend
    if backend == "auto":
        backend = "reference" if config.reference else "command"
    if config.foundation_command:
        command = _render_command(config.foundation_command, {
            "prompt": config.prompt or config.portrait_prompt, "reference": config.reference or "",
            "audio": config.audio or "", "output": output, "duration": duration,
            "width": config.width, "height": config.height, "fps": config.fps, "models": config.models_root,
        })
        _run(command, cwd=PROJECT_ROOT, env=env)
        if not output.is_file():
            raise PipelineError(f"Foundation command completed without creating {output}")
        backend = "external-foundation"
    elif config.reference:
        _make_image_video(config.reference, output, duration=duration, width=config.width, height=config.height, fps=config.fps)
        if config.prompt:
            config.notes.append("Stage 1 used the supplied portrait; prompt retained as downstream conditioning metadata.")
    elif config.allow_fallback:
        fallback = work / "fallback_portrait.png"
        _make_portrait_card(fallback)
        _make_image_video(fallback, output, duration=duration, width=config.width, height=config.height, fps=config.fps)
        backend = "synthetic-fallback"
        config.notes.append("Stage 1 fallback is a synthetic card, not a foundation model generation.")
    else:
        raise PipelineError("Prompt-only Stage 1 needs --foundation-command (Wan/Hunyuan/ComfyUI adapter).")
    return StageArtifact("foundation", output, backend, duration, {
        "prompt": config.prompt, "model_family": "Wan2.1/HunyuanVideo via external adapter" if backend == "external-foundation" else None,
    })


def _echo_command(config: PipelineConfig, output_dir: Path, duration: float) -> list[str]:
    repo = config.repos_root / "EchoMimicV3"
    script = repo / "infer_flash.py"
    model_root = config.models_root / "EchoMimicV3"
    flash_root = model_root / "flash"
    flash_hf_root = model_root / "echomimicv3-flash-pro"
    model = next((candidate for candidate in (model_root / "Wan2.1-Fun-V1.1-1.3B-InP", flash_root / "Wan2.1-Fun-V1.1-1.3B-InP", flash_hf_root / "Wan2.1-Fun-V1.1-1.3B-InP") if candidate.is_dir()), model_root / "Wan2.1-Fun-V1.1-1.3B-InP")
    transformer = next((candidate for candidate in (model_root / "transformer" / "diffusion_pytorch_model.safetensors", flash_root / "transformer" / "diffusion_pytorch_model.safetensors", flash_hf_root / "diffusion_pytorch_model.safetensors") if candidate.is_file()), model_root / "transformer" / "diffusion_pytorch_model.safetensors")
    audio_candidates = (model_root / "chinese-wav2vec2-base", flash_root / "chinese-wav2vec2-base", flash_hf_root / "chinese-wav2vec2-base", model_root / "wav2vec2-base-960h")
    wav2vec = next((candidate for candidate in audio_candidates if (candidate / "model.safetensors").is_file() and _valid_safetensors(candidate / "model.safetensors")), model_root / "wav2vec2-base-960h")
    if not script.is_file():
        raise PipelineError(f"EchoMimicV3 source is missing: {script}")
    if config.reference is None:
        raise PipelineError("EchoMimicV3 requires --reference because it animates a portrait image.")
    command = [
        sys.executable, str(script), "--image_path", str(config.reference), "--audio_path", str(config.audio),
        "--prompt", config.prompt or config.portrait_prompt, "--num_inference_steps", str(config.echo_steps),
        "--config_path", str(repo / "config" / "config.yaml"), "--model_name", str(model),
        "--transformer_path", str(transformer), "--save_path", str(output_dir), "--wav2vec_model_dir", str(wav2vec),
        "--sampler_name", "Flow_Unipc", "--video_length", str(max(1, int(duration * 25))),
        "--guidance_scale", "6", "--audio_guidance_scale", "2.5", "--seed", str(config.seed),
        "--GPU_memory_mode", config.echo_memory_mode,
        "--weight_dtype", "bfloat16", "--sample_size", str(config.echo_sample_size), str(config.echo_sample_size), "--fps", "25",
    ]
    # EchoMimic's upstream TeaCache defaults to skipping five steps, which is
    # invalid for tiny smoke renders. Keep the optimization for normal runs.
    if config.echo_steps > 5:
        command[command.index("--GPU_memory_mode"):command.index("--GPU_memory_mode")] = [
            "--enable_teacache", "--teacache_threshold", "0.1",
        ]
    else:
        # The upstream parser enables TeaCache by default even when the flag
        # is absent; make its skip window valid for tiny smoke runs.
        command[command.index("--GPU_memory_mode"):command.index("--GPU_memory_mode")] = [
            "--num_skip_start_steps", "0",
        ]
    return command


def _hallo_command(config: PipelineConfig, output: Path) -> list[str]:
    repo = config.repos_root / "Hallo"
    script = repo / "scripts" / "inference.py"
    if not script.is_file():
        raise PipelineError(f"Hallo source is missing: {script}")
    if config.reference is None:
        raise PipelineError("Hallo requires --reference because it animates a portrait image.")
    return [sys.executable, str(script), "--source_image", str(config.reference), "--driving_audio", str(config.audio), "--output", str(output)]


def _stage_two(config: PipelineConfig, foundation: StageArtifact, duration: float, work: Path, env: dict[str, str]) -> StageArtifact:
    output = work / "stage2_animated.mp4"
    backend = config.animation_backend
    if backend == "auto":
        backend = "command" if config.animation_command else "passthrough"
    if config.animation_command:
        command = _render_command(config.animation_command, {
            "foundation": foundation.path, "base_video": foundation.path, "reference": config.reference or "",
            "audio": config.audio or "", "output": output, "duration": duration, "fps": config.fps, "models": config.models_root,
        })
        _run(command, cwd=PROJECT_ROOT, env=env)
        backend = "external-animation"
    elif backend in {"echomimic", "echomimicv3"}:
        output_dir = work / "echomimic_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        _run(_echo_command(config, output_dir, duration), cwd=config.repos_root / "EchoMimicV3", env=env)
        candidates = sorted(output_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise PipelineError("EchoMimicV3 completed without an MP4 in its save directory.")
        _copy_file(candidates[0], output)
    elif backend == "hallo":
        _run(_hallo_command(config, output), cwd=config.repos_root / "Hallo", env=env)
    elif backend in {"liveportrait", "mimicmotion"}:
        raise PipelineError(f"{backend} is motion-video driven; provide --animation-command with a driving-video adapter.")
    elif backend in {"passthrough", "none"}:
        _copy_file(foundation.path, output)
        config.notes.append("Stage 2 passthrough used: no audio-driven model weights were available.")
    else:
        raise PipelineError(f"Unknown animation backend: {backend}")
    if not output.is_file():
        raise PipelineError(f"Animation stage did not create {output}")
    _release_torch_memory()
    return StageArtifact("animation", output, backend, min(duration, _probe_duration(output)), {
        "audio_conditioned": backend in {"echomimic", "echomimicv3", "hallo", "external-animation"},
        "micro_expression_model": backend in {"echomimic", "echomimicv3", "hallo"},
    })


def _stage_three(config: PipelineConfig, animated: StageArtifact, duration: float, work: Path, env: dict[str, str]) -> StageArtifact:
    enhanced = work / "stage3_enhanced.mp4"
    backend = config.enhancer
    codeformer = config.repos_root / "CodeFormer"
    if backend == "auto":
        codeformer_ready = (
            (codeformer / "inference_codeformer.py").is_file()
            and (codeformer / "weights" / "CodeFormer" / "codeformer.pth").is_file()
        )
        backend = "codeformer" if codeformer_ready else "passthrough"
    if config.enhancer_command:
        command = _render_command(config.enhancer_command, {"input": animated.path, "video": animated.path, "output": enhanced, "models": config.models_root, "duration": duration})
        _run(command, cwd=PROJECT_ROOT, env=env)
    elif backend == "codeformer":
        script = codeformer / "inference_codeformer.py"
        weights = codeformer / "weights" / "CodeFormer" / "codeformer.pth"
        if not script.is_file() or not weights.is_file():
            if config.enhancer == "codeformer":
                raise PipelineError("CodeFormer requested but source or weights are missing; run setup.sh --download-models.")
            backend = "passthrough"
            _copy_file(animated.path, enhanced)
            config.notes.append("CodeFormer weights unavailable; Stage 3 used passthrough.")
        else:
            result_dir = work / "codeformer_results"
            # CodeFormer writes one output frame per input frame. Preserve the
            # animation stream rate here; the final mux is where we promote
            # the result to the requested delivery FPS.
            animation_fps = _probe_fps(animated.path)
            _run([
                sys.executable, str(script), "--input_path", str(animated.path), "--output_path", str(result_dir),
                "--detection_model", "retinaface_resnet50", "--upscale", "1", "-w", "0.7",
                "--save_video_fps", str(animation_fps),
            ], cwd=codeformer, env=env)
            candidates = sorted(result_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not candidates:
                raise PipelineError("CodeFormer completed without an MP4 result.")
            _copy_file(candidates[0], enhanced)
    elif backend in {"passthrough", "none"}:
        _copy_file(animated.path, enhanced)
    else:
        raise PipelineError(f"Unknown enhancer: {backend}")
    return StageArtifact("enhancement", enhanced, backend, duration, {"face_restoration": backend == "codeformer"})


def _final_mux(config: PipelineConfig, enhanced: StageArtifact, duration: float) -> Path:
    output = config.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    vf = f"scale={config.width}:{config.height}:force_original_aspect_ratio=increase,crop={config.width}:{config.height},fps={config.fps},format=yuv420p"
    _run([
        "ffmpeg", "-y", "-i", str(enhanced.path), "-i", str(config.audio), "-map", "0:v:0", "-map", "1:a:0",
        "-t", f"{duration:.3f}", "-vf", vf, "-r", str(config.fps), "-c:v", "libx264", "-preset", "medium",
        "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest",
        "-movflags", "+faststart", str(output),
    ])
    return output


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    if config.reference is not None:
        config.reference = config.reference.expanduser().resolve()
    if config.audio is not None:
        config.audio = config.audio.expanduser().resolve()
    duration = _validate_input(config)
    env = _configure_runtime()
    config.models_root = config.models_root.expanduser().resolve()
    config.output = config.output.expanduser().resolve()
    config.output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(config.work_root).expanduser().resolve() if config.work_root else Path(tempfile.mkdtemp(prefix="ugc-pipeline-", dir=str(config.output.parent)))
    work.mkdir(parents=True, exist_ok=True)
    print(f"[pipeline] work directory: {work}")
    start = time.monotonic()
    foundation = _stage_one(config, duration, work, env)
    _release_torch_memory()
    animated = _stage_two(config, foundation, duration, work, env)
    enhanced = _stage_three(config, animated, duration, work, env)
    output = _final_mux(config, enhanced, duration)
    report = {
        "output": str(output), "duration_seconds": duration, "resolution": [config.width, config.height], "fps": config.fps,
        "stages": [asdict(x) for x in (foundation, animated, enhanced)], "elapsed_seconds": round(time.monotonic() - start, 3),
        "notes": config.notes, "runtime": {"arch": os.uname().machine, "cuda_allocator": os.environ.get("PYTORCH_CUDA_ALLOC_CONF")},
    }
    report = json.loads(json.dumps(report, default=str))
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not config.keep_intermediates and config.work_root is None:
        shutil.rmtree(work, ignore_errors=True)
    print(f"[done] {output}")
    print(f"[done] report: {report_path}")
    return report


def _parse_args(argv: Iterable[str] | None = None) -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Generate a local, modular UGC talking-avatar video.")
    source = parser.add_argument_group("source")
    source.add_argument("--reference", type=Path, help="High-resolution portrait image")
    source.add_argument("--prompt", help="Prompt for an external Wan/Hunyuan/ComfyUI foundation adapter")
    source.add_argument("--audio", type=Path, help="Voice audio file")
    source.add_argument("--demo", action="store_true", help="Create a 10-second synthetic integration demo")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUTS / "ugc_actor.mp4")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--foundation-backend", choices=("auto", "reference", "command"), default="auto")
    parser.add_argument("--animation-backend", choices=("auto", "passthrough", "echomimic", "echomimicv3", "hallo", "liveportrait", "mimicmotion"), default="auto")
    parser.add_argument("--enhancer", choices=("auto", "passthrough", "codeformer"), default="auto")
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--repos-root", type=Path, default=PROJECT_ROOT / "third_party")
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--foundation-command", help="Command template with {prompt} {reference} {output} {models} placeholders")
    parser.add_argument("--animation-command", help="Command template with {foundation} {reference} {audio} {output} {models} placeholders")
    parser.add_argument("--enhancer-command", help="Command template with {input} {output} {models} placeholders")
    parser.add_argument("--echo-sample-size", type=int, choices=(256, 384, 512, 768), default=512,
                        help="EchoMimic square render size; 512 is the DGX Spark-safe default")
    parser.add_argument("--echo-steps", type=int, default=8,
                        help="EchoMimic denoising steps (lower for smoke tests)")
    parser.add_argument("--echo-memory-mode", choices=("sequential_cpu_offload", "model_cpu_offload", "none"),
                        default="sequential_cpu_offload")
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--keep-intermediates", action="store_true")
    args = parser.parse_args(argv)
    if args.demo:
        args.duration = 10.0
        args.allow_fallback = True
        args.animation_backend = "passthrough"
        args.enhancer = "passthrough"
        demo_dir = PROJECT_ROOT / "outputs" / "demo_inputs"
        demo_dir.mkdir(parents=True, exist_ok=True)
        args.reference = demo_dir / "synthetic_portrait.png"
        args.audio = demo_dir / "synthetic_voice.wav"
        if not args.reference.exists():
            _make_portrait_card(args.reference)
        if not args.audio.exists():
            _make_demo_audio(args.audio, 10.0)
    return PipelineConfig(**vars(args))


def main(argv: Iterable[str] | None = None) -> int:
    try:
        run_pipeline(_parse_args(argv))
        return 0
    except PipelineError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
