from pathlib import Path

import run_seedance_pipeline as seedance


def test_build_request_uses_reference_audio_and_vertical_defaults(tmp_path):
    image = tmp_path / "portrait.jpg"
    audio = tmp_path / "voice.wav"
    image.write_bytes(b"fake-image")
    audio.write_bytes(b"fake-audio")
    args = seedance._parse_args(["--reference", str(image), "--audio", str(audio)])

    request = seedance.build_request(args)

    assert request["model"] == seedance.DEFAULT_MODEL
    assert request["ratio"] == "9:16"
    assert request["duration"] == 5
    assert request["content"][1]["role"] == "reference_image"
    assert request["content"][2]["role"] == "reference_audio"
    assert request["content"][2]["audio_url"]["url"].startswith("data:audio/wav;base64,")


def test_build_request_accepts_registered_assets_without_reading_local_files():
    args = seedance._parse_args([
        "--reference-url", "asset://portrait-123",
        "--audio-url", "asset://audio-456",
        "--model", "dreamina-seedance-2-0-fast-260128",
    ])

    request = seedance.build_request(args)

    assert request["content"][1]["image_url"]["url"] == "asset://portrait-123"
    assert request["content"][2]["audio_url"]["url"] == "asset://audio-456"


def test_video_url_supports_provider_response_shapes():
    assert seedance._video_url({"content": {"video_url": "https://example/video.mp4"}}) == "https://example/video.mp4"
    assert seedance._video_url({"content": {"file_url": {"url": "https://example/video.mp4"}}}) == "https://example/video.mp4"
