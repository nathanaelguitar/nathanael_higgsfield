from pathlib import Path

import pytest

import run_ugc_pipeline as pipeline


def test_command_template_quotes_and_expands():
    command = pipeline._render_command(
        "runner --prompt {prompt} --out {output}",
        {"prompt": "a person says hello", "output": "/tmp/a b.mp4"},
    )
    assert command == ["runner", "--prompt", "a person says hello", "--out", "/tmp/a b.mp4"]


def test_invalid_dimensions_are_rejected(tmp_path, monkeypatch):
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"not media")
    monkeypatch.setattr(pipeline, "_require_ffmpeg", lambda: None)
    monkeypatch.setattr(pipeline, "_probe_duration", lambda _: 1.0)
    config = pipeline.PipelineConfig(reference=audio, audio=audio, width=1079)
    with pytest.raises(pipeline.PipelineError, match="dimensions"):
        pipeline._validate_input(config)


def test_demo_cli_sets_ten_seconds(monkeypatch):
    monkeypatch.setattr(pipeline, "_make_portrait_card", lambda path: path.write_bytes(b"demo"))
    monkeypatch.setattr(pipeline, "_make_demo_audio", lambda path, duration: path.write_bytes(b"demo"))
    config = pipeline._parse_args(["--demo"])
    assert config.duration == 10.0
    assert config.allow_fallback is True
    assert config.animation_backend == "passthrough"


def test_final_filter_is_vertical_1080p(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(pipeline, "_run", lambda command, **kwargs: calls.append(command))
    config = pipeline.PipelineConfig(audio=tmp_path / "voice.wav", output=tmp_path / "out.mp4", width=1080, height=1920, fps=60)
    config.audio.write_bytes(b"audio")
    enhanced = pipeline.StageArtifact("enhancement", tmp_path / "in.mp4", "passthrough", 1.0)
    enhanced.path.write_bytes(b"video")
    result = pipeline._final_mux(config, enhanced, 1.0)
    assert result == (tmp_path / "out.mp4").resolve()
    joined = " ".join(calls[-1])
    assert "scale=1080:1920" in joined
    assert "fps=60" in joined
