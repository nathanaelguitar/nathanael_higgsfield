from run_fal_seedance_text import build_arguments


def test_text_video_payload_is_vertical_and_audio_enabled() -> None:
    payload = build_arguments("dialogue", "720p", 5, "9:16")

    assert payload == {
        "prompt": "dialogue",
        "resolution": "720p",
        "duration": 5,
        "aspect_ratio": "9:16",
        "generate_audio": True,
    }
