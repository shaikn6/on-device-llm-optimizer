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
    mask: mx.array | None = None,
) -> mx.array:
    """Compute the knowledge distillation training loss.

    Args:
        logits_s: Student logits, shape [batch, seq_len, vocab_size], float32.
        logits_t: Teacher logits, shape [batch, seq_len, vocab_size], float32.
        labels: Ground-truth token ids, shape [batch, seq_len], int32.
        temperature: Softening temperature T > 1 softens distributions.
        alpha: Weight for soft (KL) loss; (1-alpha) weights hard (CE) loss.
               alpha=1.0 → pure KL; alpha=0.0 → pure CE.
        mask: Optional real-token mask, shape [batch, seq_len], float32, with
              1.0 for real tokens and 0.0 for padding. When given, padded
              positions are excluded from both loss terms instead of being
              averaged in as if they were real supervision targets.

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

    # --- Hard loss: cross-entropy against ground-truth labels ---
    logits_flat = logits_s.reshape(-1, vocab_size)                 # [B*S, V]
    labels_flat = labels.reshape(-1)                               # [B*S]
    log_probs = logits_flat - mx.logsumexp(logits_flat, axis=-1, keepdims=True)
    ce = -log_probs[mx.arange(labels_flat.shape[0]), labels_flat]  # [B*S]

    if mask is not None:
        mask_flat = mask.reshape(-1)
        denom = mx.maximum(mx.sum(mask), 1e-8)
        soft_loss = mx.sum(kl * mask) / denom * (temperature ** 2)
        hard_loss = mx.sum(ce * mask_flat) / denom
    else:
        soft_loss = mx.mean(kl) * (temperature ** 2)
        hard_loss = mx.mean(ce)

    return alpha * soft_loss + (1.0 - alpha) * hard_loss
