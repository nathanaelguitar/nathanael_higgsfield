from run_fal_voice_clone import build_arguments


def test_f5_tts_payload_omits_optional_reference_text() -> None:
    payload = build_arguments("https://audio", "say this", None, "F5-TTS")

    assert payload == {
        "gen_text": "say this",
        "ref_audio_url": "https://audio",
        "model_type": "F5-TTS",
        "remove_silence": True,
    }


def test_f5_tts_payload_accepts_reference_transcript() -> None:
    payload = build_arguments("https://audio", "say this", "reference words", "E2-TTS")

    assert payload["ref_text"] == "reference words"
    assert payload["model_type"] == "E2-TTS"
