"""CPU-only tests for StudentConfig arithmetic and validation.

These import only src.model.config (pure Python, no MLX), so they run on any
platform without Apple Silicon or model downloads.
"""
import pytest

from src.model.config import StudentConfig


def test_defaults_reproduce_design_spec():
    cfg = StudentConfig()
    assert cfg.num_layers == 12
    assert cfg.hidden_dim == 1024
    assert cfg.num_heads == 8
    assert cfg.vocab_size == 32064


def test_head_dim_and_ff_dim_derivation():
    cfg = StudentConfig()
    # 1024 / 8 = 128 per-head; 4x expansion = 4096
    assert cfg.head_dim == 128
    assert cfg.ff_dim == 4096
    assert cfg.head_dim * cfg.num_heads == cfg.hidden_dim


def test_head_dim_honours_custom_values():
    cfg = StudentConfig(hidden_dim=256, num_heads=4, ff_multiplier=2)
    assert cfg.head_dim == 64
    assert cfg.ff_dim == 512


def test_head_dim_rejects_indivisible_shape():
    # hidden_dim not divisible by num_heads must fail loudly, not silently round.
    cfg = StudentConfig(hidden_dim=100, num_heads=8)
    with pytest.raises(AssertionError):
        _ = cfg.head_dim
