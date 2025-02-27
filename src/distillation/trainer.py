"""Knowledge distillation training loop.

Loads teacher (Phi-3 Mini, frozen) and student (custom transformer), trains with
KD loss, saves checkpoints every N steps, saves final model to models/student_fp32/.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as mx_utils
from mlx_lm import load as mlx_load

from src.distillation.dataset import batch_iter, load_alpaca
from src.distillation.losses import kd_loss
from src.model.config import StudentConfig
from src.model.student import StudentModel


class DistillationTrainer:
    """Orchestrates knowledge distillation from a frozen teacher to a student model.

    Args:
        cfg: Parsed distill_config.yaml dict (see configs/distill_config.yaml).
        checkpoint_dir: Directory for mid-training checkpoints.
        output_dir: Directory to save the final FP32 student model.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        checkpoint_dir: str | Path = "checkpoints",
        output_dir: str | Path = "models/student_fp32",
    ) -> None:
        self.cfg = cfg
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir = Path(output_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load teacher (MLX INT4, frozen)
        teacher_name = cfg["teacher"]["model"]
        print(f"Loading teacher: {teacher_name} (INT4) …")
        self.teacher, self.tokenizer = mlx_load(
            teacher_name, tokenizer_config={"trust_remote_code": True}
        )
        self.teacher.eval()
        # Freeze teacher — no gradients needed
        self.teacher.freeze()

        # Build student
        s_cfg = cfg["student"]
        student_config = StudentConfig(
            num_layers=s_cfg["layers"],
            hidden_dim=s_cfg["hidden_dim"],
            num_heads=s_cfg["attention_heads"],
            vocab_size=s_cfg["vocab_size"],
        )
        print(f"Building student: {student_config} …")
        self.student = StudentModel(student_config)

        d_cfg = cfg["distillation"]
        self.temperature = float(d_cfg["temperature"])
        self.alpha = float(d_cfg["alpha"])
        self.batch_size = int(d_cfg["batch_size"])
        self.steps = int(d_cfg["steps"])
        self.checkpoint_every = int(d_cfg["checkpoint_every"])
        self.lr = float(d_cfg["lr"])
        self.max_samples = int(d_cfg["max_samples"])
        self.max_seq_len = 512  # Cap for training efficiency

        self.optimizer = optim.AdamW(learning_rate=self.lr)

    def _loss_fn(self, student: StudentModel, tokens: mx.array) -> mx.array:
        """Compute KD loss for one batch. Called by mx.value_and_grad."""
        logits_s = student(tokens)                            # [B, S, V]
        with mx.no_grad():
            logits_t = self.teacher(tokens)                   # [B, S, V] — frozen
        labels = tokens[:, 1:]                                # next-token labels
        logits_s = logits_s[:, :-1, :]                       # align
        logits_t = logits_t[:, :-1, :]
        return kd_loss(logits_s, logits_t, labels, self.temperature, self.alpha)

    def _save_checkpoint(self, step: int) -> None:
        path = self.checkpoint_dir / f"step_{step:06d}"
        path.mkdir(exist_ok=True)
        flat_weights = dict(mx_utils.tree_flatten(self.student.parameters()))
        mx.savez(str(path / "weights.npz"), **flat_weights)
        print(f"  Checkpoint saved → {path}")

    def _save_final(self) -> None:
        flat_weights = dict(mx_utils.tree_flatten(self.student.parameters()))
        mx.savez(str(self.output_dir / "weights.npz"), **flat_weights)
        config_path = self.output_dir / "config.json"
        config_path.write_text(json.dumps(self.student.cfg.__dict__, indent=2))
        print(f"Final model saved → {self.output_dir}")

    def train(self) -> None:
        """Run the full distillation training loop."""
        print("Loading dataset …")
        train_tokens, val_tokens = load_alpaca(
            tokenizer_name=self.cfg["teacher"]["model"],
            max_samples=self.max_samples,
            max_seq_len=self.max_seq_len,
        )
        print(f"  Train: {len(train_tokens):,} | Val: {len(val_tokens):,}")

        loss_and_grad = nn.value_and_grad(self.student, self._loss_fn)
        step = 0
        start_time = time.time()

        while step < self.steps:
            for batch in batch_iter(train_tokens, self.batch_size, self.max_seq_len):
                loss, grads = loss_and_grad(self.student, batch)
                self.optimizer.update(self.student, grads)
                mx.eval(self.student.parameters(), self.optimizer.state)

                step += 1
                if step % 50 == 0:
                    elapsed = time.time() - start_time
                    print(f"  Step {step:>6}/{self.steps}  loss={float(loss):.4f}  elapsed={elapsed:.0f}s")
                if step % self.checkpoint_every == 0:
                    self._save_checkpoint(step)
                if step >= self.steps:
                    break

        self._save_final()
        print("Training complete.")
