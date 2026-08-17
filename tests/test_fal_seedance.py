from __future__ import annotations

import argparse

from run_fal_seedance import build_arguments, estimate_cost


def test_build_arguments_uses_fal_reference_fields() -> None:
    payload = build_arguments("say this", "https://image", "https://audio", "720p", 5, "9:16")

    assert payload == {
        "prompt": "say this",
        "image_urls": ["https://image"],
        "audio_urls": ["https://audio"],
        "resolution": "720p",
        "duration": 5,
        "aspect_ratio": "9:16",
        "generate_audio": True,
    }


def test_build_arguments_supports_video_reference_without_image() -> None:
    payload = build_arguments(
        "say this",
        None,
        "https://audio",
        "720p",
        5,
        "9:16",
        video_url="https://video",
    )

    assert payload["video_urls"] == ["https://video"]
    assert "image_urls" not in payload


def test_fast_720p_estimate() -> None:
    assert estimate_cost(5, "720p") == 1.2095


def test_unknown_endpoint_has_no_estimate() -> None:
    assert estimate_cost(5, "720p", "custom/model") is None
