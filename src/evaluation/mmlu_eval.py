"""MMLU 5-shot evaluation for language model benchmarking.

MMLU (Massive Multitask Language Understanding) measures accuracy across 57 subjects.
We sample `n` questions uniformly at random, format as 5-shot prompts, and score by
selecting the answer choice (A/B/C/D) with highest log-probability.

Dataset: cais/mmlu, config "all", split "test".
"""
from __future__ import annotations

import random

import mlx.core as mx
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from src.model.student import StudentModel

_CHOICES = ["A", "B", "C", "D"]

_5_SHOT_PREAMBLE = """\
The following are multiple choice questions (with answers) about various topics.

Q: What is the capital of France?
A) Rome B) Berlin C) Paris D) Madrid
Answer: C

Q: Which planet is closest to the Sun?
A) Venus B) Mercury C) Earth D) Mars
Answer: B

Q: What is 15 × 4?
A) 50 B) 55 C) 60 D) 65
Answer: C

Q: Water's chemical formula?
A) H2O2 B) HO C) H2O D) OH
Answer: C

Q: Who wrote Hamlet?
A) Dickens B) Tolstoy C) Shakespeare D) Austen
Answer: C

"""


def _format_question(row: dict) -> tuple[str, int]:
    """Format one MMLU row into a prompt string and the correct choice index (0-3)."""
    choices = row["choices"]
    formatted = (
        f"Q: {row['question']}\n"
        f"A) {choices[0]} B) {choices[1]} C) {choices[2]} D) {choices[3]}\n"
        "Answer:"
    )
    return _5_SHOT_PREAMBLE + formatted, int(row["answer"])


def mmlu_accuracy(
    model: StudentModel,
    tokenizer: PreTrainedTokenizerBase,
    n: int = 200,
    seed: int = 42,
) -> float:
    """Estimate model accuracy on a random sample of n MMLU questions.

    Args:
        model: StudentModel or any MLX module returning [batch, seq, vocab] logits.
        tokenizer: Tokenizer matching the model's vocabulary.
        n: Number of MMLU questions to evaluate.
        seed: Random seed for question sampling.

    Returns:
        Accuracy as a float in [0, 1]. Multiply by 100 for percentage.
    """
    ds = load_dataset("cais/mmlu", "all", split="test")
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))

    model.eval()
    correct = 0
    choice_ids = [tokenizer.encode(f" {c}", add_special_tokens=False)[0] for c in _CHOICES]

    for idx in indices:
        prompt, answer_idx = _format_question(ds[idx])
        token_ids = tokenizer.encode(prompt, truncation=True, max_length=1024)
        tokens = mx.array([token_ids], dtype=mx.int32)
        logits = model(tokens)                                  # [1, T, V]
        last_logits = logits[0, -1, :]                         # [V] — after "Answer:"
        scores = mx.array([float(last_logits[cid]) for cid in choice_ids])
        pred = int(mx.argmax(scores))
        if pred == answer_idx:
            correct += 1

    return correct / len(indices)
