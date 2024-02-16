"""Knowledge distillation loss: weighted KL divergence (soft) + cross-entropy (hard).

Loss = alpha * soft_loss + (1 - alpha) * hard_loss

Where:
  soft_loss = KL( softmax(S/T) || softmax(Teacher/T) ) scaled by T²
  hard_loss = CrossEntropy(S, ground-truth labels)

Reference: Hinton et al. "Distilling the Knowledge in a Neural Network" (2015).
"""
import mlx.core as mx


def kd_loss(
    logits_s: mx.array,
    logits_t: mx.array,
    labels: mx.array,
    temperature: float,
    alpha: float,
) -> mx.array:
    """Compute the knowledge distillation training loss.

    Args:
        logits_s: Student logits, shape [batch, seq_len, vocab_size], float32.
        logits_t: Teacher logits, shape [batch, seq_len, vocab_size], float32.
        labels: Ground-truth token ids, shape [batch, seq_len], int32.
        temperature: Softening temperature T > 1 softens distributions.
        alpha: Weight for soft (KL) loss; (1-alpha) weights hard (CE) loss.
               alpha=1.0 → pure KL; alpha=0.0 → pure CE.

    Returns:
        Scalar loss value (0-d array).
    """
    vocab_size = logits_s.shape[-1]

    # --- Soft loss: KL divergence at temperature T ---
    s_soft = mx.softmax(logits_s / temperature, axis=-1)          # [B, S, V]
    t_soft = mx.softmax(logits_t / temperature, axis=-1)          # [B, S, V]
    # KL(P_s || P_t) = sum P_s * (log P_s - log P_t)
    log_s = mx.log(s_soft + 1e-8)
    log_t = mx.log(t_soft + 1e-8)
    kl = mx.sum(s_soft * (log_s - log_t), axis=-1)                # [B, S]
    soft_loss = mx.mean(kl) * (temperature ** 2)                  # scale by T²

    # --- Hard loss: cross-entropy against ground-truth labels ---
    logits_flat = logits_s.reshape(-1, vocab_size)                 # [B*S, V]
    labels_flat = labels.reshape(-1)                               # [B*S]
    log_probs = logits_flat - mx.logsumexp(logits_flat, axis=-1, keepdims=True)
    hard_loss = -mx.mean(log_probs[mx.arange(labels_flat.shape[0]), labels_flat])

    return alpha * soft_loss + (1.0 - alpha) * hard_loss
