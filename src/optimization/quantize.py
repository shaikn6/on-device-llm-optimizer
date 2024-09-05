"""MLX INT4 quantization of a saved StudentModel weights file.

Quantization strategy:
  - Load FP32 weights from weights.npz
  - For each Linear weight matrix: apply group-wise INT4 quantization
    using mlx.core.quantize (returns quantized weights + scales + biases)
  - Save to output_dir/weights.npz in the quantized format
  - Copy config.json unchanged

Group-wise quantization splits each weight row into groups of `group_size`
values, computes per-group scale and zero-point, and stores 4-bit integers.
This matches mlx-lm's own quantization convention for compatibility.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import mlx.core as mx
import mlx.utils as mx_utils


def quantize_int4(
    fp32_dir: str | Path,
    out_dir: str | Path,
    group_size: int = 64,
) -> None:
    """Quantize a saved FP32 StudentModel to INT4 (group-wise).

    Args:
        fp32_dir: Directory containing weights.npz and config.json (FP32).
        out_dir: Destination directory for quantized weights.npz and config.json.
        group_size: Number of values per quantization group (must be ≥ 32).
    """
    fp32_dir = Path(fp32_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load FP32 weights: mx.load returns dict[str, mx.array] for .npz
    weights: dict[str, mx.array] = mx.load(str(fp32_dir / "weights.npz"))

    quantized: dict[str, mx.array] = {}
    for name, tensor in weights.items():
        # Only quantize 2-D weight matrices (Linear layers); keep everything else FP32
        if tensor.ndim == 2 and tensor.shape[-1] % group_size == 0:
            # mx.quantize returns (quantized_weights, scales, biases) — 4-bit packed
            q_w, scales, biases = mx.quantize(tensor, bits=4, group_size=group_size)
            quantized[name] = q_w
            quantized[f"{name}_scales"] = scales
            quantized[f"{name}_biases"] = biases
        else:
            quantized[name] = tensor

    mx.savez(str(out_dir / "weights.npz"), **quantized)
    # Copy config.json verbatim
    shutil.copy(fp32_dir / "config.json", out_dir / "config.json")
    print(f"INT4 model saved → {out_dir}")
