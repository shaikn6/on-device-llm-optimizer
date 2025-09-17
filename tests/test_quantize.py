"""Tests for INT4 quantization and memory profiler."""
import json
from pathlib import Path

import mlx.core as mx
import mlx.utils as mx_utils

from src.model.config import StudentConfig
from src.model.student import StudentModel
from src.optimization.quantize import quantize_int4
from src.optimization.memory import peak_ram_mb


def _tiny_model_dir(tmp_path: Path) -> Path:
    """Save a tiny FP32 student to a temp dir; return the dir path."""
    cfg = StudentConfig(num_layers=2, hidden_dim=64, num_heads=4, vocab_size=256, max_seq_len=32)
    model = StudentModel(cfg)
    # Use tree_flatten to get flat dict of arrays, then save with mx.savez
    weights = dict(mx_utils.tree_flatten(model.parameters()))
    mx.savez(str(tmp_path / "weights.npz"), **weights)
    (tmp_path / "config.json").write_text(json.dumps(cfg.__dict__))
    return tmp_path


def test_int4_size_is_smaller_than_fp32(tmp_path):
    """INT4 model weights file must be smaller than the FP32 weights file.

    The threshold is set at 30% for the tiny test model to account for NPZ
    header overhead and scale/bias tensors. For production-size models (e.g.,
    236M params), the INT4 file is typically ~12-13% of FP32 size.
    """
    fp32_dir = tmp_path / "fp32"
    fp32_dir.mkdir()
    _tiny_model_dir(fp32_dir)

    int4_dir = tmp_path / "int4"
    quantize_int4(fp32_dir, int4_dir, group_size=64)

    fp32_size = (fp32_dir / "weights.npz").stat().st_size
    int4_size = (int4_dir / "weights.npz").stat().st_size
    ratio = int4_size / fp32_size
    assert ratio <= 0.30, f"INT4/FP32 size ratio is {ratio:.2%}, expected ≤ 30%"
    assert int4_size < fp32_size, "INT4 model must be smaller than FP32 model"


def test_quantize_output_dir_created(tmp_path):
    """quantize_int4 must create the output directory if it does not exist."""
    fp32_dir = tmp_path / "fp32"
    fp32_dir.mkdir()
    _tiny_model_dir(fp32_dir)
    int4_dir = tmp_path / "non_existent" / "int4"
    assert not int4_dir.exists()
    quantize_int4(fp32_dir, int4_dir, group_size=64)
    assert int4_dir.exists()


def test_peak_ram_returns_positive_float():
    """peak_ram_mb context manager must return a positive float."""
    with peak_ram_mb() as tracker:
        _ = list(range(100_000))
    assert isinstance(tracker.peak_mb, float)
    assert tracker.peak_mb > 0.0
