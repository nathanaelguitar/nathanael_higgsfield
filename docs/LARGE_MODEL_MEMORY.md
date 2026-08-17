# Large-model memory findings

## DGX Spark test result

The workstation exposes approximately 121 GiB of unified host/device memory.
The local EchoMimic workflow remains usable, but the current Hunyuan/Wan
foundation-video paths are not safe with the full-length 480P portrait test.

### Wan 2.1 I2V 14B

- Checkpoint tested: `Wan-AI/Wan2.1-I2V-14B-480P`.
- Download size: approximately 76 GiB.
- The official path needed a PyTorch SDPA fallback because FlashAttention
  wheels are not available for this ARM64 environment.
- The unquantized offload run loaded the model but consumed essentially the
  entire unified-memory pool before generation and was stopped.

### HunyuanVideo 1.5

- Checkpoint tested: 480P I2V step-distilled Diffusers package.
- Transformer size: 8.331B parameters.
- Package size: approximately 32 GiB including transformer, text encoders,
  VAE, and vision encoder.
- An 81-frame, eight-step test was attempted with CPU offload, then with the
  full bf16 pipeline resident on CUDA, and finally with TorchAO int8
  weight-only quantization across 786 Linear layers.
- All three configurations reached OOM-level unified-memory pressure at the
  start of video denoising. The process was stopped before a usable Hunyuan
  output was written.

The important distinction is that parameter storage was not the only cost.
The video denoising activations, temporary copies, text/vision conditioning,
and unified-memory page migration dominate the peak. CPU offload is not a free
optimization on a unified-memory system because host and device pages compete
for the same physical pool.

## NVFP4 follow-up

NVFP4 is worth testing later, but it must be treated as a separate, measured
backend rather than enabled globally. The next experiment should:

1. Start from a cold, measured system with no unrelated model processes.
2. Quantize only the Hunyuan transformer; keep VAE, text encoders, and vision
   conditioning in bf16 unless a validated kernel supports them.
3. Use a 4–8 second clip and the lowest supported frame count first.
4. Record `MemAvailable`, swap use, and CUDA allocation before loading,
   immediately before denoising, and after every inference step.
5. Abort at a fixed safety floor instead of waiting for the kernel OOM path.

TorchAO int8 was validated at the operator level on this machine. TorchAO
int4 was not usable in the current environment because its required `mslk`
kernel was unavailable. No NVFP4 generation has been claimed or run yet.

## Current recommendation

Use local models for portrait preparation, audio preprocessing, FFmpeg, and
post-processing. Use Seedance for the foundation/motion generation until a
quantized Hunyuan implementation demonstrates a stable peak-memory profile.
