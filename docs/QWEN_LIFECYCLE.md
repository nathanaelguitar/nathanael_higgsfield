# Qwen lifecycle on DGX Spark

The local Qwen service is intentionally kept separate from the UGC runtime and
TRIBE. It runs in Docker so the model can be stopped before another large model
uses the unified memory pool.

## Current service

- Container: `qwen3.8-27b-sglang`
- Image: `lmsysorg/sglang:qwen38-27b`
- Model: `RadixArk/Qwen3.8-27B-NVFP4`
- API: `http://127.0.0.1:8888`
- Model cache: `/home/nathanaelguitar/miaai-qwen38-27b-sglang/.cache/huggingface`
- Triton cache: `/home/nathanaelguitar/miaai-qwen38-27b-sglang/.cache/triton`

The container has restart policy `no`, so it will not come back by itself after
being stopped or after a reboot.

## Stop before TRIBE or video inference

Use Docker's normal stop path so SGLang receives `SIGTERM`, drains active
requests, and releases its CUDA allocations:

```bash
docker stop --timeout 30 qwen3.8-27b-sglang
```

Confirm that no SGLang process remains and that the GPU allocation is gone:

```bash
pgrep -af 'sglang|Qwen3.8-27B|qwen3.8-27b' || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader,nounits
```

The stopped container and Hugging Face cache should be preserved. Do not run
`docker rm` or delete the cache when the goal is only to free memory.

## Start it again

The normal restart does not redownload the model:

```bash
docker start qwen3.8-27b-sglang
until curl -fsS http://127.0.0.1:8888/v1/models >/dev/null; do
  sleep 5
done
curl -fsS http://127.0.0.1:8888/v1/models
```

The first request after restart may take time while SGLang restores the model
into memory. Check logs if the readiness loop does not finish:

```bash
docker logs -f --tail 100 qwen3.8-27b-sglang
```

## Recreate only if the container was removed

The following reproduces the current configuration. It reuses the existing
cache and does not delete model files:

```bash
docker run -d \
  --name qwen3.8-27b-sglang \
  --gpus all \
  --ipc host \
  --network host \
  -v /home/nathanaelguitar/miaai-qwen38-27b-sglang/.cache/huggingface:/root/.cache/huggingface \
  -v /home/nathanaelguitar/miaai-qwen38-27b-sglang/.cache/triton:/root/.triton \
  lmsysorg/sglang:qwen38-27b \
  python3 -m sglang.launch_server \
  --model-path RadixArk/Qwen3.8-27B-NVFP4 \
  --served-model-name qwen3.8-27b-sglang \
  --trust-remote-code \
  --mem-fraction-static 0.60 \
  --attention-backend flashinfer \
  --chunked-prefill-size 8192 \
  --disable-prefill-cuda-graph \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-ssm-dtype bfloat16 \
  --mamba-full-memory-ratio 4.21 \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --max-mamba-cache-size 80 \
  --max-running-requests 10 \
  --context-length 262144 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --sampling-defaults model \
  --host 0.0.0.0 \
  --port 8888
```

Do not start Qwen concurrently with TRIBE, Wan, HunyuanVideo, or another large
CUDA workload on the Spark. Stop it first and verify the allocation is clear.
