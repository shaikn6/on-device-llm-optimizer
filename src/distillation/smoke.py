"""Small, real knowledge-distillation run used to produce a *measured* number.

The full Phi-3 -> 236M run needs hours and a ~4 GB teacher. This module holds the
pieces of a much smaller version (tiny student, small open teacher, a few hundred
steps) so that a held-out perplexity can be measured, not assumed. It reuses the
repo's own ``StudentModel``, ``kd_loss`` and ``DistillationTrainer._loss_fn``.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from src.distillation.trainer import DistillationTrainer


class Float32Logits:
    """Wrap a (possibly bf16) teacher so the KD loss sees float32 logits."""

    def __init__(self, model: Callable[[mx.array], mx.array]) -> None:
        self.model = model

    def __call__(self, tokens: mx.array) -> mx.array:
        return self.model(tokens).astype(mx.float32)


def split_windows(
    token_ids: list[int], seq_len: int, val_frac: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Split one long token stream into a train stream and non-overlapping val windows.

    The split is contiguous (train = first part, val = last part), so no val text is
    seen in training. Returns (train_stream, val_windows[N, seq_len]).
    """
    ids = np.asarray(token_ids, dtype=np.int32)
    cut = int(len(ids) * (1.0 - val_frac))
    train, val = ids[:cut], ids[cut:]
    n_val = len(val) // seq_len
    return train, val[: n_val * seq_len].reshape(n_val, seq_len)


def train_batches(train: np.ndarray, batch_size: int, seq_len: int, seed: int):
    """Endless generator of random [batch, seq_len] windows from the train stream."""
    rng = np.random.RandomState(seed)
    hi = len(train) - seq_len
    while True:
        starts = rng.randint(0, hi, size=batch_size)
        yield mx.array(np.stack([train[s : s + seq_len] for s in starts]), dtype=mx.int32)


def held_out_ppl(model: Callable[[mx.array], mx.array], val: np.ndarray, batch_size: int = 16) -> float:
    """exp(mean next-token cross-entropy) over every val window (lower is better)."""
    total_nll, total_tokens = 0.0, 0
    for i in range(0, len(val), batch_size):
        tokens = mx.array(val[i : i + batch_size], dtype=mx.int32)
        logits = model(tokens)[:, :-1, :].astype(mx.float32)
        targets = tokens[:, 1:]
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        picked = mx.take_along_axis(logp, targets[..., None], axis=-1)
        total_nll += -float(mx.sum(picked))
        total_tokens += targets.size
    return math.exp(total_nll / total_tokens)


def distill(
    student: nn.Module,
    teacher: Callable[[mx.array], mx.array],
    batches,
    steps: int,
    lr: float,
    temperature: float,
    alpha: float,
) -> list[float]:
    """Train ``student`` for ``steps`` steps with the repo's KD loss; return per-step loss.

    ``alpha=0.0`` is plain next-token cross-entropy (no teacher signal) and serves as
    the control that shows how much of the gain comes from the teacher.
    """
    # _loss_fn only reads teacher / temperature / alpha from self.
    fake_self = SimpleNamespace(teacher=teacher, temperature=temperature, alpha=alpha)

    def loss_fn(model, tokens, mask):
        return DistillationTrainer._loss_fn(fake_self, model, tokens, mask)

    loss_and_grad = nn.value_and_grad(student, loss_fn)
    optimizer = optim.AdamW(learning_rate=lr)
    losses: list[float] = []
    for _, tokens in zip(range(steps), batches):
        mask = mx.ones(tokens.shape, dtype=mx.float32)
        loss, grads = loss_and_grad(student, tokens, mask)
        optimizer.update(student, grads)
        mx.eval(student.parameters(), optimizer.state)
        losses.append(float(loss))
    return losses
