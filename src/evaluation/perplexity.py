"""Perplexity computation on a held-out text corpus.

Perplexity = exp( -1/N * sum_i log P(token_i | context) )

Lower perplexity = better language model fit to the distribution.
"""
from __future__ import annotations

import math

import mlx.core as mx
from transformers import PreTrainedTokenizerBase

from src.model.student import StudentModel


def compute_perplexity(
    model: StudentModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    max_seq_len: int = 512,
) -> float:
    """Compute average perplexity of a model on a list of texts.

    Args:
        model: A StudentModel (or any MLX module with __call__(tokens) → logits).
        tokenizer: Tokenizer matching the model's vocabulary.
        texts: List of plain text strings to evaluate on.
        max_seq_len: Sequences are truncated to this length.

    Returns:
        Perplexity as a Python float. Lower is better.
    """
    total_log_prob = 0.0
    total_tokens = 0

    model.eval()
    for text in texts:
        token_ids = tokenizer.encode(text, truncation=True, max_length=max_seq_len)
        if len(token_ids) < 2:
            continue
        tokens = mx.array([token_ids], dtype=mx.int32)         # [1, T]
        logits = model(tokens)                                  # [1, T, V]
        # Shift: predict token t from context [0..t-1]
        log_probs = logits[0, :-1, :]                          # [T-1, V]
        log_probs = log_probs - mx.logsumexp(log_probs, axis=-1, keepdims=True)
        targets = mx.array(token_ids[1:], dtype=mx.int32)      # [T-1]
        token_log_probs = log_probs[mx.arange(len(targets)), targets]
        total_log_prob += float(mx.sum(token_log_probs))
        total_tokens += len(targets)

    if total_tokens == 0:
        return float("inf")
    avg_nll = -total_log_prob / total_tokens
    return math.exp(avg_nll)
