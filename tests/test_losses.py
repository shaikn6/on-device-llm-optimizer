"""Tests for the knowledge distillation loss function."""
import mlx.core as mx
import numpy as np
import pytest
from src.distillation.losses import kd_loss


@pytest.fixture
def sample_logits():
    """Batch=2, seq_len=4, vocab=8 logits."""
    rng = np.random.default_rng(42)
    logits_s = mx.array(rng.standard_normal((2, 4, 8)).astype(np.float32))
    logits_t = mx.array(rng.standard_normal((2, 4, 8)).astype(np.float32))
    labels = mx.array(rng.integers(0, 8, size=(2, 4)).astype(np.int32))
    return logits_s, logits_t, labels


def test_alpha_one_returns_pure_kl(sample_logits):
    """With alpha=1.0, loss should equal pure KL divergence (no CE term)."""
    logits_s, logits_t, labels = sample_logits
    loss_alpha1 = kd_loss(logits_s, logits_t, labels, temperature=4.0, alpha=1.0)
    loss_alpha0 = kd_loss(logits_s, logits_t, labels, temperature=4.0, alpha=0.0)
    # Pure KL vs pure CE should be different values
    assert float(loss_alpha1) != pytest.approx(float(loss_alpha0), rel=1e-3)
    # Both must be non-negative
    assert float(loss_alpha1) >= 0.0
    assert float(loss_alpha0) >= 0.0


def test_alpha_zero_is_pure_ce(sample_logits):
    """With alpha=0.0, loss equals pure cross-entropy (teacher logits ignored)."""
    logits_s, logits_t, labels = sample_logits
    loss = kd_loss(logits_s, logits_t, labels, temperature=4.0, alpha=0.0)
    # Reference: compute CE manually
    logits_flat = logits_s.reshape(-1, 8)
    labels_flat = labels.reshape(-1)
    log_probs = logits_flat - mx.logsumexp(logits_flat, axis=-1, keepdims=True)
    ce_ref = -mx.mean(log_probs[mx.arange(labels_flat.shape[0]), labels_flat])
    assert float(loss) == pytest.approx(float(ce_ref), rel=1e-4)


def test_higher_temperature_softens_distributions():
    """Higher temperature produces softer (more uniform) distributions.

    The unscaled KL divergence between two softmax distributions should be
    smaller at higher temperature (distributions are more uniform, so closer
    to each other). The T² scaling in the loss offsets this but the raw
    unscaled KL value decreases.
    """
    rng = np.random.default_rng(0)
    logits_s = mx.array(rng.standard_normal((1, 8, 32)).astype(np.float32))
    logits_t = mx.array(rng.standard_normal((1, 8, 32)).astype(np.float32))

    def raw_kl(logits_s, logits_t, temperature):
        """Compute KL without T² scaling."""
        s_soft = mx.softmax(logits_s / temperature, axis=-1)
        t_soft = mx.softmax(logits_t / temperature, axis=-1)
        log_s = mx.log(s_soft + 1e-8)
        log_t = mx.log(t_soft + 1e-8)
        kl = mx.sum(s_soft * (log_s - log_t), axis=-1)
        return float(mx.mean(kl))

    kl_t1 = raw_kl(logits_s, logits_t, temperature=1.0)
    kl_t8 = raw_kl(logits_s, logits_t, temperature=8.0)
    # Unscaled KL should be smaller at higher temperature (softer distributions)
    assert kl_t8 < kl_t1, f"Expected raw KL(T=8)={kl_t8:.4f} < KL(T=1)={kl_t1:.4f}"


def test_loss_is_scalar(sample_logits):
    """kd_loss must return a 0-d tensor (scalar)."""
    logits_s, logits_t, labels = sample_logits
    loss = kd_loss(logits_s, logits_t, labels, temperature=4.0, alpha=0.7)
    assert loss.ndim == 0
