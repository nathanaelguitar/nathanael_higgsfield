"""Minimal torchtune compatibility surface required by NeuCodec.

NeuCodec imports RotaryPositionalEmbeddings from torchtune, while recent
torchtune eagerly imports optional quantization modules that are incompatible
with the DGX Spark torch/torchao combination. Keeping this small implementation
ahead of site-packages avoids importing those unrelated modules.
"""

from __future__ import annotations

import torch
from torch import nn


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10_000) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        theta = 1.0 / (base ** (torch.arange(0, dim, 2).float()[: dim // 2] / dim))
        self.register_buffer("theta", theta, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, max_seq_len: int) -> None:
        positions = torch.arange(max_seq_len, dtype=self.theta.dtype, device=self.theta.device)
        angles = torch.einsum("i,j->ij", positions, self.theta).float()
        self.register_buffer("cache", torch.stack((angles.cos(), angles.sin()), dim=-1), persistent=False)

    def forward(self, x: torch.Tensor, *, input_pos: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.size(1)
        cache = self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        shaped = x.float().reshape(*x.shape[:-1], -1, 2)
        cache = cache.view(-1, shaped.size(1), 1, shaped.size(3), 2)
        result = torch.stack(
            (shaped[..., 0] * cache[..., 0] - shaped[..., 1] * cache[..., 1],
             shaped[..., 1] * cache[..., 0] + shaped[..., 0] * cache[..., 1]),
            dim=-1,
        )
        return result.flatten(3).type_as(x)


__all__ = ["RotaryPositionalEmbeddings"]
