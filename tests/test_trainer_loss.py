"""Smoke test for DistillationTrainer._loss_fn (no teacher download, no training run)."""
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import mlx.utils as mx_utils
import pytest

pytest.importorskip("mlx_lm")  # trainer.py imports mlx_lm at module level

from src.distillation.trainer import DistillationTrainer
from src.model.config import StudentConfig
from src.model.student import StudentModel


def _tiny(seed: int) -> StudentModel:
    mx.random.seed(seed)
    cfg = StudentConfig(num_layers=1, hidden_dim=32, num_heads=4, vocab_size=64, max_seq_len=16)
    return StudentModel(cfg)


def test_loss_fn_runs_and_grads_flow_to_student_only():
    teacher = _tiny(1)
    teacher.freeze()
    student = _tiny(2)
    # _loss_fn only reads teacher/temperature/alpha from self.
    fake_self = SimpleNamespace(teacher=teacher, temperature=4.0, alpha=0.7)
    tokens = mx.array([[1, 5, 7, 9, 2, 3], [4, 6, 8, 10, 11, 0]], dtype=mx.int32)
    mask = mx.array([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0]], dtype=mx.float32)

    loss_and_grad = nn.value_and_grad(student, lambda s, t, m: DistillationTrainer._loss_fn(fake_self, s, t, m))
    loss, grads = loss_and_grad(student, tokens, mask)
    mx.eval(loss, grads)

    assert loss.shape == () and bool(mx.isfinite(loss))
    flat = dict(mx_utils.tree_flatten(grads))
    assert flat and any(float(mx.abs(g).sum()) > 0 for g in flat.values())
