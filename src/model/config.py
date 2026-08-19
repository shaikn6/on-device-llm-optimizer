"""Hyperparameter dataclass for the student transformer."""
from dataclasses import dataclass


@dataclass
class StudentConfig:
    """All architectural hyperparameters for the student model.

    Default values reproduce the ~236M parameter target from the design spec.
    """
    num_layers: int = 12
    hidden_dim: int = 1024
    num_heads: int = 8
    vocab_size: int = 32064
    max_seq_len: int = 2048
    # Feed-forward expansion factor (standard 4× for transformers)
    ff_multiplier: int = 4
    # Dropout — only active during training
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        assert self.hidden_dim % self.num_heads == 0
        return self.hidden_dim // self.num_heads

    @property
    def ff_dim(self) -> int:
        return self.hidden_dim * self.ff_multiplier
