# Seedance 2.0 API handoff

Seedance is the next recommended backend for this project after the local
foundation-model tests exceeded the DGX Spark's unified-memory budget. The
fal.ai hosted API is the preferred US-compatible path; the BytePlus ModelArk
API remains documented below for completeness.
The reference-to-video path accepts reference images, video clips, and audio;
the prompt can describe spoken dialogue and the service returns an MP4 with
native synchronized audio. For the CanopyChat test, use the same portrait and
the prepared five-second voice track.

## BytePlus ModelArk (authenticated)

The BytePlus key was created in the default ModelArk project and is stored
locally, outside the repository, at:

```text
/home/nathanaelguitar/.config/ugc_actor_engine/byteplus.env
```

Load it only into the current shell:

```bash
set -a
source /home/nathanaelguitar/.config/ugc_actor_engine/byteplus.env
set +a
```

BytePlus uses the `ARK_API_KEY` environment variable and the Southeast Asia
base URL:

```text
https://ark.ap-southeast.bytepluses.com/api/v3
```

The local client defaults to `seedance-1-5-pro-251215`, which is the better
first choice for this project because it supports native audio/video sync and
can be used with usage billing after account/model activation. Seedance 2.0
model IDs are `dreamina-seedance-2-0-260128` and
`dreamina-seedance-2-0-fast-260128`.

Important billing distinction: BytePlus currently requires a Seedance 2.0
resource pack to activate the 2.0 family. After activation, additional usage
can fall back to pay-as-you-go. That is not the same as a monthly subscription,
but it also means 2.0 is not a pure zero-commitment API test. Seedance 1.5 Pro
is the usage-based first test path.

API-key management page:

<https://console.byteplus.com/ark/region:ark+ap-southeast-1/apikey>

## Recommended endpoint

For this account, start with fal.ai's usage-billed fast endpoint:

```text
bytedance/seedance-2.0/fast/reference-to-video
```

Install its small client without installing any CUDA model stack:

```bash
./setup.sh --install-fal
export FAL_KEY='your-key-in-the-current-shell-only'
.venv/bin/python run_fal_seedance.py \
  --reference outputs/canopychat_inputs/reference_close.jpg \
  --audio outputs/canopychat_inputs/voice_5s.wav \
  --output outputs/fal_seedance_canopychat_5s.mp4
```

The published fast 720p rate is approximately $0.2419 per generated second,
so a five-second first pass is about $1.21 before any account-specific taxes or
rate changes. The endpoint accepts the portrait and audio as provider-uploaded
files and returns a video with generated audio. Keep `FAL_KEY` out of Git and
chat transcripts.

Official model page and API schema: [fal.ai Seedance 2.0 Fast reference-to-video](https://fal.ai/models/bytedance/seedance-2.0/fast/reference-to-video).

### BytePlus endpoints

Use the standard quality endpoint first:

```text
bytedance/seedance-2.0/reference-to-video
```

The fast tier is a useful lower-cost fallback:

```text
bytedance/seedance-2.0/fast/reference-to-video
```

Start with `duration: "5"`, `resolution: "720p"`, and `aspect_ratio: "9:16"`.
Use 1080p only after the 720p composition and identity are acceptable.

## Authentication

Complete the BytePlus account profile, model activation, and API-key creation
in the browser. The agent will not submit payment, purchase a resource pack,
or enable unbounded pay-as-you-go billing on your behalf. Do not paste the key
into the repository, a prompt, or a chat transcript.

For the fal.ai alternative, create an API-scope key and paste the complete
`key_id:key_secret` value into the private file
`/home/nathanaelguitar/.config/ugc_actor_engine/fal.env`:

```bash
FAL_KEY=key_id:key_secret
```

fal only shows the secret at creation time. The official guidance recommends
the API scope for calling ready-to-use models; an ADMIN key is unnecessary.

Verify only that the variable is present; never print its value:

```bash
test -n "${FAL_KEY:-}" && echo "FAL_KEY is set"
```

The repository ignores `.env` files and generated model/media directories.
For a persistent local setup, use a user-owned secret manager or an ignored
`.env` file loaded by the shell.

## Local client

The repository includes a standard-library client that uploads small local
inputs as data URLs, submits one asynchronous task, polls it, and downloads the
MP4 without touching CUDA memory:

```bash
.venv/bin/python run_seedance_pipeline.py \
  --reference outputs/canopychat_inputs/reference_close.jpg \
  --audio outputs/canopychat_inputs/voice_5s.wav \
  --model seedance-1-5-pro-251215 \
  --output outputs/seedance_canopychat_5s.mp4
```

Use `--dry-run` to validate the request without calling ModelArk. For Seedance
2.0, registered authorized human assets should be supplied as
`--reference-url asset://...` and `--audio-url asset://...` when required by
the provider's real-person asset policy.

## Python request shape

Install the provider client in an isolated environment or the project venv:

```bash
uv pip install --python .venv/bin/python fal-client
```

The first request should be small and explicit:

```python
import fal_client

result = fal_client.subscribe(
    "bytedance/seedance-2.0/reference-to-video",
    arguments={
        "prompt": (
            "Use [Image1] as the same UGC creator. Keep her identity, hairstyle, "
            "clothing, and vertical selfie framing. She looks into the lens and "
            "says exactly: \"That's why you should switch to CanopyChat.\" "
            "Natural eye blinks, small head movement, realistic facial motion, "
            "clean mouth articulation, no captions, no watermark."
        ),
        "image_urls": ["https://.../reference_close.jpg"],
        "audio_urls": ["https://.../voice_5s.wav"],
        "resolution": "720p",
        "duration": "5",
        "aspect_ratio": "9:16",
        "generate_audio": True,
    },
)

print(result["video"]["url"])
```

The provider needs publicly reachable or provider-uploaded media URLs. Local
paths such as `outputs/canopychat_inputs/reference_close.jpg` cannot be sent
directly. The integration should upload the portrait and WAV/MP3 to the
provider's file endpoint, submit the request, download the returned MP4, and
then run the existing FFmpeg validation/mux stage.

### Separate voice-clone pass

When Seedance's native generated voice is not close enough, use the authorized
original voice sample with fal F5-TTS, then use that exact output as the audio
reference for Seedance 2.0 and mux it back over the returned video:

```bash
.venv/bin/python run_fal_voice_clone.py \
  --reference-audio outputs/canopychat_inputs/original_voice_reference_8s.wav \
  --text "That's why you should switch to CanopyChat." \
  --output outputs/canopychat_inputs/fal_voice_clone_canopychat.wav
```

F5-TTS supports zero-shot cloning from one sample and automatic reference-audio
transcription. Use this only for a voice for which the speaker has authorized
the synthesis. See the [official F5-TTS API](https://fal.ai/models/fal-ai/f5-tts/api).

## Prompt and input policy

- Use one clear reference portrait for identity and one clean voice track for
  the first test.
- Put the spoken line in double quotes and say `says exactly`.
- Do not ask the model to clone a voice unless the speaker has given informed
  consent and the use is authorized.
- Treat the returned URL as temporary; copy the result into `outputs/` only
  after the request succeeds.
- Label synthetic media where required by the distribution channel.

## Acceptance checks

The Seedance adapter should fail fast unless all of the following are true:

1. `FAL_KEY` is set without logging its value.
2. The reference image and audio upload successfully.
3. The API returns a video URL and the downloaded file passes `ffprobe`.
4. The result is five seconds, vertical 9:16, and contains a video and audio
   stream.
5. A human review confirms identity continuity, mouth timing, eye contact,
   and absence of unwanted captions or watermarks.

## Official references

- [Seedance 2.0 reference-to-video API](https://fal.ai/models/bytedance/seedance-2.0/reference-to-video)
- [Seedance 2.0 API overview and endpoint schema](https://fal.ai/docs/model-api-reference/video-generation-api/bytedance-seedance-2.0-text-to-video)
- [ByteDance Seedance 2.0 overview](https://seed.bytedance.com/en/seedance2_0)
