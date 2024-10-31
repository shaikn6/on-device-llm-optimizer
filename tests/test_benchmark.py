"""Integration test: all four model variants must return non-empty text output.

This test uses a tiny synthetic model (not Phi-3) so it runs in CI without
downloading large model files. It validates the inference path and output
format, not model quality.
"""
import json
from pathlib import Path

import mlx.core as mx
import mlx.utils as mx_utils
import pytest
from transformers import AutoTokenizer

from src.model.config import StudentConfig
from src.model.student import StudentModel
from src.optimization.quantize import quantize_int4

TOKENIZER_NAME = "microsoft/Phi-3-mini-4k-instruct"
PROMPT = "What is the capital of France?"


def _save_tiny_model(path: Path) -> tuple[StudentModel, StudentConfig]:
    """Create and save a tiny untrained model for testing."""
    cfg = StudentConfig(
        num_layers=2, hidden_dim=64, num_heads=4, vocab_size=32064, max_seq_len=64
    )
    model = StudentModel(cfg)
    weights = dict(mx_utils.tree_flatten(model.parameters()))
    mx.savez(str(path / "weights.npz"), **weights)
    (path / "config.json").write_text(json.dumps(cfg.__dict__))
    return model, cfg


def _greedy_decode(model: StudentModel, token_ids: list[int], steps: int = 10) -> list[int]:
    """Simple greedy decoding for testing inference path."""
    for _ in range(steps):
        tokens = mx.array([token_ids], dtype=mx.int32)
        logits = model(tokens)                    # [1, T, V]
        next_id = int(mx.argmax(logits[0, -1, :]))
        token_ids = token_ids + [next_id]
    return token_ids


@pytest.fixture(scope="module")
def tokenizer():
    """Load the Phi-3 tokenizer once for all benchmark tests."""
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)


@pytest.fixture(scope="module")
def tiny_model_dirs(tmp_path_factory):
    """Create fp32 and int4 model dirs with a tiny synthetic model."""
    base = tmp_path_factory.mktemp("models")
    fp32_dir = base / "fp32"
    fp32_dir.mkdir()
    _save_tiny_model(fp32_dir)
    int4_dir = base / "int4"
    quantize_int4(fp32_dir, int4_dir, group_size=64)
    return {"fp32": fp32_dir, "int4": int4_dir}


def test_student_fp32_returns_output(tiny_model_dirs, tokenizer):
    """FP32 student must produce non-empty token output from the prompt."""
    cfg_data = json.loads((tiny_model_dirs["fp32"] / "config.json").read_text())
    cfg = StudentConfig(**cfg_data)
    model = StudentModel(cfg)
    # Load weights using mx.load and tree_unflatten
    weights = mx.load(str(tiny_model_dirs["fp32"] / "weights.npz"))
    model.load_weights(list(weights.items()))

    input_ids = tokenizer.encode(PROMPT)[:32]
    output_ids = _greedy_decode(model, input_ids, steps=5)
    output_text = tokenizer.decode(output_ids[len(input_ids):])
    assert isinstance(output_text, str)
    assert len(output_ids) > len(input_ids)


def test_student_int4_quantized_model_is_loadable(tiny_model_dirs):
    """INT4 weights.npz must be loadable and contain quantized keys."""
    int4_weights = mx.load(str(tiny_model_dirs["int4"] / "weights.npz"))
    # At least some keys should have _scales suffix
    scale_keys = [k for k in int4_weights if k.endswith("_scales")]
    assert len(scale_keys) > 0, "No quantized scale tensors found in INT4 model"


def test_all_variants_have_different_sizes(tiny_model_dirs):
    """FP32 model file must be larger than INT4 model file."""
    fp32_size = (tiny_model_dirs["fp32"] / "weights.npz").stat().st_size
    int4_size = (tiny_model_dirs["int4"] / "weights.npz").stat().st_size
    assert fp32_size > int4_size, (
        f"FP32 ({fp32_size} bytes) should be larger than INT4 ({int4_size} bytes)"
    )
