"""Tiny-model checks for the KD smoke-run helpers (no downloads)."""
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

pytest.importorskip("mlx_lm")  # smoke.py imports the trainer, which imports mlx_lm

from src.distillation.smoke import distill, held_out_ppl, split_windows, train_batches
from src.model.config import StudentConfig
from src.model.student import StudentModel

VOCAB, SEQ = 64, 16


def _tiny(seed: int) -> StudentModel:
    mx.random.seed(seed)
    cfg = StudentConfig(num_layers=1, hidden_dim=32, num_heads=4, vocab_size=VOCAB, max_seq_len=SEQ)
    return StudentModel(cfg)


def test_split_windows_is_disjoint_and_shaped():
    train, val = split_windows(list(range(1000)), seq_len=SEQ, val_frac=0.1)
    assert val.shape == (6, SEQ)  # 100 val tokens // 16 windows
    assert train.max() < val.min()  # contiguous split: no val token seen in training


def test_distillation_loss_is_finite_and_decreases():
    teacher, student = _tiny(1), _tiny(2)
    teacher.freeze()
    rng = np.random.RandomState(0)
    stream = rng.randint(0, VOCAB, size=2000).astype(np.int32)
    val = stream[:SEQ * 8].reshape(8, SEQ)
    before = held_out_ppl(student, val)
    losses = distill(student, teacher, train_batches(stream, 8, SEQ, seed=0),
                     steps=30, lr=1e-2, temperature=4.0, alpha=0.7)
    after = held_out_ppl(student, val)
    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0]
    assert np.isfinite(before) and np.isfinite(after)


def test_committed_results_file_is_labelled_smoke_run():
    path = Path(__file__).resolve().parents[1] / "results" / "kd_smoke.json"
    data = json.loads(path.read_text())
    assert "SMOKE RUN" in data["kind"]
    ppl = data["val_perplexity"]
    assert ppl["student_kd"] < ppl["student_before"]
