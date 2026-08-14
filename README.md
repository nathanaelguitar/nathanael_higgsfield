# Local UGC actor engine

This is a stage-isolated orchestration layer for a local NVIDIA DGX Spark. It
can run a 10-second FFmpeg integration demo immediately, while model stages are
enabled only when their repositories and weights are present.

## Architecture

1. Foundation: reference portrait passthrough, or an external Wan/HunyuanVideo/
   ComfyUI command supplied with `--foundation-command`.
2. Animation: EchoMimicV3 or Hallo for audio-driven lips and expressions. A
   custom command can wrap LivePortrait or MimicMotion when a driving video or
   pose stream is available.
3. Enhancement/mux: optional CodeFormer command, then FFmpeg crops to 1080x1920,
   encodes H.264/AAC, and muxes the input voice.

Each stage runs separately and releases PyTorch CUDA caches before the next
stage. Sequential CPU offload is passed to EchoMimicV3 to reduce peak memory
pressure on unified memory.

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

External commands receive shell-quoted placeholders `{prompt}`, `{reference}`,
`{output}`, `{duration}`, `{width}`, `{height}`, `{fps}`, and `{models}`.
Only use portraits and voices with informed consent and appropriate rights;
label synthetic media where required by the distribution channel.
