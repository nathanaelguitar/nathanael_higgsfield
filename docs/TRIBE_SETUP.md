# TRIBE v2 local setup

TRIBE is kept in its own `.tribe_venv` so its dependency set does not alter the
UGC video environment. The setup and download scripts are deliberately
download-only: they do not call `TribeModel.from_pretrained`, construct a
feature extractor, or allocate CUDA model memory.

## Setup

From the repository root:

```bash
./scripts/setup_tribe.sh
```

The public assets downloaded by the default command are:

- `models/tribev2/best.ckpt` — TRIBE prediction checkpoint;
- `models/tribev2/config.yaml` — matching configuration;
- `models/tribev2/encoders/facebook_vjepa2-vitg-fpc64-256` — V-JEPA2-G;
- `models/tribev2/encoders/facebook_w2v-bert-2.0` — Wav2Vec2-BERT.

The encoder files are large, so the downloader uses one worker and resumes
partial Hugging Face files after a network interruption.

## Gated Llama encoder

The released TRIBE configuration references `meta-llama/Llama-3.2-3B`. It is a
gated Hugging Face repository and must be authorized for the logged-in account.
After accepting access for that model, resume only the missing asset with:

```bash
./scripts/setup_tribe.sh --include-llama
```

The downloader ignores the repository policy document and fetches the model
configuration, tokenizer, and weight shards needed by Transformers. It never
prints the Hugging Face token.

Do not substitute a different text encoder: the TRIBE checkpoint was trained
with the configured encoder features, so changing the encoder requires a
separate compatibility experiment.

## Verify without loading models

```bash
.tribe_venv/bin/python - <<'PY'
import torch
import tribev2

print("tribev2 import: ok")
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("No TribeModel was constructed.")
PY

find models/tribev2 -name '*.incomplete' -print
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader,nounits
```

The `find` command should produce no output after successful downloads, and
`nvidia-smi` should show no TRIBE process. The checkpoint and encoders remain on
disk until an explicit inference run.

## Memory discipline

Stop Qwen first using [`QWEN_LIFECYCLE.md`](QWEN_LIFECYCLE.md). Run one TRIBE
candidate at a time, keep the public encoder downloads separate from the video
models, and write extracted features to disk. Do not load TRIBE, Qwen, Wan, or
HunyuanVideo concurrently on the Spark until an isolated memory measurement
has been completed.
