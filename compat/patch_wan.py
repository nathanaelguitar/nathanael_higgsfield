#!/usr/bin/env python3
"""Make the official Wan2.1 attention path work without FlashAttention wheels."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_wan.py /path/to/Wan2.1")
    attention = Path(sys.argv[1]) / "wan" / "modules" / "attention.py"
    if not attention.is_file():
        raise SystemExit(f"Wan attention module not found: {attention}")
    source = attention.read_text(encoding="utf-8")
    marker = "    half_dtypes = (torch.float16, torch.bfloat16)\n"
    if "# DGX-Spark SDPA fallback" in source:
        return 0
    fallback = """    # DGX-Spark SDPA fallback: FlashAttention wheels are not published for
    # every aarch64/CUDA combination. Keep the official API and use PyTorch's
    # fused scaled-dot-product attention instead.
    if not FLASH_ATTN_2_AVAILABLE and not FLASH_ATTN_3_AVAILABLE:
        out_dtype = q.dtype
        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)
        if q_scale is not None:
            q = q * q_scale
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=causal,
            dropout_p=dropout_p,
        )
        return out.transpose(1, 2).contiguous().type(out_dtype)

"""
    if marker not in source:
        raise SystemExit("Wan attention source changed; fallback patch was not applied")
    attention.write_text(source.replace(marker, fallback + marker, 1), encoding="utf-8")
    print(f"patched {attention}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
