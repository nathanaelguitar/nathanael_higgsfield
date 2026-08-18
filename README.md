# Local UGC actor engine

This is a stage-isolated orchestration layer for a local NVIDIA DGX Spark. It
can run a 10-second FFmpeg integration demo immediately, and it has a tested
real EchoMimicV3 audio-driven path when its repositories and weights are present.

## Architecture

1. Foundation: reference portrait passthrough, or an external Wan/HunyuanVideo/
   ComfyUI command supplied with `--foundation-command`.
2. Animation: EchoMimicV3 or Hallo for audio-driven lips and expressions. A
   custom command can wrap LivePortrait or MimicMotion when a driving video or
   pose stream is available.
3. Enhancement/mux: optional CodeFormer command, then FFmpeg crops to 1080x1920,
   encodes H.264/AAC, and muxes the input voice.

Each stage runs separately and releases PyTorch CUDA caches before the next
stage. Sequential CPU offload is applied to EchoMimicV3 (the upstream flag did
not activate it) to reduce peak memory pressure on unified memory. The default
512px/8-step profile is intended for production portraits; use 256px/2-step
for a fast smoke test.

## Setup and demo

```bash
./setup.sh
.venv/bin/python run_ugc_pipeline.py --demo --output outputs/demo.mp4
.venv/bin/python -m pytest -q
```

The demo uses a synthetic portrait card and generated tone. It validates the
CLI, stage handoff, vertical 1080p/60fps encode, and audio mux; it is not a
claim of model-quality talking-head synthesis.

For the heavier stack, use `./setup.sh --install-heavy` and download weights
only after confirming storage and licenses: `./setup.sh --download-models`.

```bash
.venv/bin/python run_ugc_pipeline.py \
  --reference portrait.jpg --audio voice.wav \
  --animation-backend echomimic --enhancer auto \
  --output outputs/creator.mp4
```

### Authorized local voice cloning

Install the isolated NeuTTS environment so its speech dependencies do not
disturb the video runtime:

```bash
./setup.sh --install-voice-clone
```

Provide a clean reference recording and its exact transcript. The pipeline
normalizes the recording to the 24 kHz mono format required by the codec,
synthesizes `--script`, and feeds the result into the same animation and mux
stages:

```bash
.venv/bin/python run_ugc_pipeline.py \
  --reference portrait.jpg \
  --script "That's why you should switch to CanopyChat." \
  --voice-reference authorized_voice.wav \
  --voice-reference-text "Exact words spoken in authorized_voice.wav" \
  --animation-backend echomimic --enhancer codeformer \
  --output outputs/creator-cloned-voice.mp4
```

Only use reference voices with the speaker's informed consent and the rights
to synthesize them; label synthetic media where required.

For a bounded DGX Spark smoke render:

```bash
.venv/bin/python run_ugc_pipeline.py \
  --reference portrait.jpg --audio voice.wav --duration 10 \
  --animation-backend echomimic --enhancer codeformer \
  --echo-sample-size 256 --echo-steps 2 --output outputs/creator.mp4
```

The verified local 10-second smoke path completed in about 95 seconds on the
DGX Spark and emitted 1080x1920 H.264/AAC at 60 fps. The sample used a
synthetic portrait and low-step settings, so it validates orchestration and
audio conditioning—not photorealistic quality. Use a real high-resolution
portrait and the default 512px/8-step profile for quality, subject to runtime
and memory limits. CodeFormer is optional and falls back only when `--enhancer
auto` is used; `--enhancer codeformer` fails fast if its weights are absent.

External commands receive shell-quoted placeholders `{prompt}`, `{reference}`,
`{output}`, `{duration}`, `{width}`, `{height}`, `{fps}`, and `{models}`.
Only use portraits and voices with informed consent and appropriate rights;
label synthetic media where required by the distribution channel.

### Large-model and hosted backends

The local Wan 2.1 14B and HunyuanVideo 1.5 experiments are documented in
[`docs/LARGE_MODEL_MEMORY.md`](docs/LARGE_MODEL_MEMORY.md). The recommended
next quality experiment is the hosted Seedance 2.0 reference-to-video path
through fal.ai. It is usage-billed and does not load a local video model into
the DGX Spark memory pool. See [`docs/SEEDANCE_API.md`](docs/SEEDANCE_API.md)
for authentication, request shape, media-upload requirements, and acceptance
checks.

The TRIBE-based pre-finalization scoring hypothesis and resource-safe experiment
plan are documented in [`docs/TRIBE_VIRALITY_HYPOTHESIS.md`](docs/TRIBE_VIRALITY_HYPOTHESIS.md).
The affective-region ranking CLI is [`scripts/predict_virality.py`](scripts/predict_virality.py).
The paired founder/faceless hook scripts are in
[`docs/CANOPYCHAT_AB_TEST_SCRIPTS.md`](docs/CANOPYCHAT_AB_TEST_SCRIPTS.md).

The isolated TRIBE environment, encoder downloads, gated Llama resume path,
and no-load verification are documented in
[`docs/TRIBE_SETUP.md`](docs/TRIBE_SETUP.md).

Qwen memory lifecycle and restart instructions are documented in
[`docs/QWEN_LIFECYCLE.md`](docs/QWEN_LIFECYCLE.md). Stop Qwen before loading
TRIBE or any large local video model.
