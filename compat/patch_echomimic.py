#!/usr/bin/env python3
"""Apply the small DGX-Spark compatibility patch to EchoMimicV3.

EchoMimic's flash script exposes CPU-offload flags but, in the upstream
version, always moves the complete pipeline to the device.  This patch keeps
the repository unmodified in git while making setup reproducible locally.
"""

from __future__ import annotations

import sys
from pathlib import Path


def patch(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if "enable_sequential_cpu_offload" in text:
        updated = text.replace(
            'parser.add_argument("--enable_teacache", action="store_true", default=True, help="Enable TeaCache")',
            'parser.add_argument("--enable_teacache", action="store_true", default=False, help="Enable TeaCache")',
            1,
        )
        if updated != text:
            path.write_text(updated, encoding="utf-8")
        return True
    original = text

    text = text.replace(
        "import decord\n",
        "try:\n    import decord  # optional on aarch64; unused by flash inference\nexcept ImportError:\n    decord = None\n",
        1,
    )
    text = text.replace(
        'parser.add_argument("--enable_teacache", action="store_true", default=True, help="Enable TeaCache")',
        'parser.add_argument("--enable_teacache", action="store_true", default=False, help="Enable TeaCache")',
        1,
    )
    text = text.replace(
        "    pipeline.to(device=device)\n\n    coefficients = get_teacache_coefficients(model_name)",
        "    if GPU_memory_mode == \"sequential_cpu_offload\":\n"
        "        pipeline.enable_sequential_cpu_offload(device=device)\n"
        "    elif GPU_memory_mode == \"model_cpu_offload\":\n"
        "        pipeline.enable_model_cpu_offload(device=device)\n"
        "    else:\n"
        "        pipeline.to(device=device)\n\n"
        "    coefficients = get_teacache_coefficients(model_name)",
        1,
    )
    text = text.replace(
        "    pipeline.to(device=device)\n\n    # Create output directory",
        "    if GPU_memory_mode not in (\"sequential_cpu_offload\", \"model_cpu_offload\"):\n"
        "        pipeline.to(device=device)\n\n"
        "    # Create output directory",
        1,
    )

    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return "enable_sequential_cpu_offload" in text


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} /path/to/EchoMimicV3", file=sys.stderr)
        return 2
    target = Path(sys.argv[1]) / "infer_flash.py"
    if not target.is_file():
        print(f"EchoMimicV3 infer script not found: {target}", file=sys.stderr)
        return 2
    if not patch(target):
        print("EchoMimicV3 patch did not match the expected upstream layout", file=sys.stderr)
        return 1
    print(f"Patched {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
