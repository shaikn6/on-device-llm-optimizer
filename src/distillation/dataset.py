"""Alpaca 52K dataset loader and tokenization for distillation training.

Produces batches of token tensors with shape [batch_size, max_seq_len].
Uses the Phi-3 tokenizer so student and teacher share the same vocabulary.
"""
from __future__ import annotations

import random
from typing import Iterator

import mlx.core as mx
from datasets import load_dataset
from transformers import AutoTokenizer

ALPACA_DATASET = "tatsu-lab/alpaca"
PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)


def _format_example(row: dict) -> str:
    """Format one Alpaca row into a single prompt+response string."""
    return PROMPT_TEMPLATE.format(
        instruction=row.get("instruction", ""),
        input=row.get("input", ""),
        output=row.get("output", ""),
    )


def load_alpaca(
    tokenizer_name: str,
    max_samples: int,
    max_seq_len: int,
    train_frac: float = 0.95,
    seed: int = 42,
) -> tuple[list[list[int]], list[list[int]]]:
    """Download Alpaca 52K, tokenize, and split into train/val.

    Args:
        tokenizer_name: HuggingFace model name for the tokenizer
                        (e.g. "microsoft/Phi-3-mini-4k-instruct").
        max_samples: Cap on number of examples to use.
        max_seq_len: Maximum token sequence length; examples are truncated.
        train_frac: Fraction of data for training (rest is validation).
        seed: Random seed for the split.

    Returns:
        (train_tokens, val_tokens): Lists of token-id lists, one per example.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    ds = load_dataset(ALPACA_DATASET, split="train")

    rng = random.Random(seed)
    examples = [_format_example(row) for row in ds]
    rng.shuffle(examples)
    examples = examples[:max_samples]

    token_lists = [
        tokenizer.encode(text, truncation=True, max_length=max_seq_len)
        for text in examples
    ]

    split_idx = int(len(token_lists) * train_frac)
    return token_lists[:split_idx], token_lists[split_idx:]


def batch_iter(
    token_lists: list[list[int]],
    batch_size: int,
    max_seq_len: int,
    pad_id: int = 0,
    shuffle: bool = True,
    seed: int = 0,
) -> Iterator[mx.array]:
    """Yield batches of padded token arrays.

    Args:
        token_lists: List of token-id lists from load_alpaca().
        batch_size: Number of examples per batch.
        max_seq_len: Sequences are truncated then padded to this length.
        pad_id: Token id used for padding.
        shuffle: Whether to shuffle at the start of each epoch.
        seed: Random seed for shuffling.

    Yields:
        mx.array of shape [batch_size, max_seq_len], dtype int32.
    """
    rng = random.Random(seed)
    indices = list(range(len(token_lists)))
    if shuffle:
        rng.shuffle(indices)

    for start in range(0, len(indices) - batch_size + 1, batch_size):
        batch_ids = indices[start : start + batch_size]
        batch = []
        for idx in batch_ids:
            toks = token_lists[idx][:max_seq_len]
            padded = toks + [pad_id] * (max_seq_len - len(toks))
            batch.append(padded)
        yield mx.array(batch, dtype=mx.int32)
