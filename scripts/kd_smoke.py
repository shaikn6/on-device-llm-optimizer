"""Small, real KD run: tiny student, small open teacher, measured held-out perplexity.

Usage:
    python scripts/kd_smoke.py [--steps 500] [--out results/kd_smoke.json]

The defaults are the configuration of the committed results/kd_smoke.json.

This is a SMOKE RUN, not the Phi-3 -> 236M design run. It writes the measured numbers
to a JSON file; nothing in it is estimated.

Teacher: HuggingFaceTB/SmolLM2-135M (Apache-2.0), revision pinned below.
Text:    Tiny Shakespeare (public-domain text; file from karpathy/char-rnn, MIT),
         downloaded and checked against a SHA-256.
"""
import argparse
import hashlib
import json
import platform
import time
import urllib.request
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from huggingface_hub import snapshot_download
from mlx_lm import load as mlx_load

from src.distillation.smoke import (
    Float32Logits,
    distill,
    held_out_ppl,
    split_windows,
    train_batches,
)
from src.model.config import StudentConfig
from src.model.student import StudentModel

TEACHER = "HuggingFaceTB/SmolLM2-135M"
TEACHER_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
TEXT_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
TEXT_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"


def fetch_text(cache: Path) -> str:
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(TEXT_URL, cache)
    raw = cache.read_bytes()
    if hashlib.sha256(raw).hexdigest() != TEXT_SHA256:
        raise ValueError(f"{cache} does not match the pinned SHA-256")
    return raw.decode("utf-8")


def make_student(vocab_size: int, args: argparse.Namespace, seed: int) -> StudentModel:
    mx.random.seed(seed)
    cfg = StudentConfig(
        num_layers=args.layers, hidden_dim=args.hidden_dim, num_heads=args.heads,
        vocab_size=vocab_size, max_seq_len=args.seq_len,
    )
    model = StudentModel(cfg)
    mx.eval(model.parameters())
    return model


def n_params(model) -> int:
    return sum(v.size for _, v in tree_flatten(model.parameters()))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-cache", default="data/tinyshakespeare.txt")
    p.add_argument("--out", default="results/kd_smoke.json")
    args = p.parse_args()

    text = fetch_text(Path(args.data_cache))
    teacher_dir = snapshot_download(TEACHER, revision=TEACHER_REVISION)
    teacher_raw, tokenizer = mlx_load(teacher_dir)
    teacher_raw.eval()
    teacher = Float32Logits(teacher_raw)
    vocab = teacher_raw.args.vocab_size

    train, val = split_windows(tokenizer.encode(text), args.seq_len)
    print(f"tokens: train={len(train):,} val={val.size:,} ({len(val)} windows of {args.seq_len})")

    result = {
        "kind": "SMOKE RUN - tiny student, few steps. NOT the Phi-3 -> 236M design-target run.",
        "teacher": {"model": TEACHER, "revision": TEACHER_REVISION, "license": "Apache-2.0",
                    "params": n_params(teacher_raw)},
        "dataset": {"name": "Tiny Shakespeare", "url": TEXT_URL, "sha256": TEXT_SHA256,
                    "license": "public-domain text (file from karpathy/char-rnn, MIT)",
                    "train_tokens": int(len(train)), "val_tokens": int(val.size)},
        "config": {k: getattr(args, k) for k in
                   ("steps", "batch_size", "seq_len", "lr", "temperature", "alpha",
                    "layers", "hidden_dim", "heads", "seed")},
        "hardware": f"{platform.machine()} {platform.system()}, mlx {mx.__version__}",
        "val_perplexity": {},
    }
    ppl = result["val_perplexity"]
    ppl["teacher"] = held_out_ppl(teacher, val)

    for name, alpha in (("student_kd", args.alpha), ("student_ce_only_control", 0.0)):
        student = make_student(vocab, args, args.seed)
        if name == "student_kd":
            result["student_params"] = n_params(student)
            ppl["student_before"] = held_out_ppl(student, val)
            print(f"student params={result['student_params']:,} before: ppl={ppl['student_before']:.1f}")
        t0 = time.time()
        losses = distill(student, teacher, train_batches(train, args.batch_size, args.seq_len, args.seed),
                         args.steps, args.lr, args.temperature, alpha)
        ppl[name] = held_out_ppl(student, val)
        result.setdefault("train_seconds", {})[name] = round(time.time() - t0, 1)
        result.setdefault("train_loss_first_last", {})[name] = [round(losses[0], 4), round(losses[-1], 4)]
        print(f"{name}: ppl={ppl[name]:.1f} loss {losses[0]:.3f}->{losses[-1]:.3f} ({time.time()-t0:.0f}s)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
