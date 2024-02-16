"""Custom 1B student transformer built in MLX nn.Module.

Architecture:
  token_embed (vocab_size → hidden_dim)
  + positional_embed (max_seq_len → hidden_dim)
  → N × TransformerBlock (pre-norm, multi-head attn, FFN with SiLU gate)
  → final LayerNorm
  → lm_head (hidden_dim → vocab_size, weight-tied to token_embed)
"""
import mlx.core as mx
import mlx.nn as nn
from src.model.config import StudentConfig


class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention with causal mask."""

    def __init__(self, cfg: StudentConfig) -> None:
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.scale = cfg.head_dim ** -0.5
        dim = cfg.hidden_dim
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)

        # Causal mask: upper triangle = -inf
        mask = mx.triu(mx.full((T, T), float("-inf")), k=1)
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale + mask
        attn = mx.softmax(attn, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """SwiGLU-style gated feed-forward network."""

    def __init__(self, cfg: StudentConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_dim, cfg.ff_dim, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_dim, cfg.ff_dim, bias=False)
        self.down_proj = nn.Linear(cfg.ff_dim, cfg.hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LayerNorm → Attn + residual → LayerNorm → FFN + residual."""

    def __init__(self, cfg: StudentConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.hidden_dim)
        self.attn = MultiHeadAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.hidden_dim)
        self.ffn = FeedForward(cfg)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class StudentModel(nn.Module):
    """Full student language model.

    Inputs: token ids of shape [batch, seq_len] (int32)
    Outputs: logits of shape [batch, seq_len, vocab_size] (float32)
    """

    def __init__(self, cfg: StudentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.hidden_dim)
        self.blocks = [TransformerBlock(cfg) for _ in range(cfg.num_layers)]
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        # Weight-tied lm_head: no separate parameter, reuses token_embed weight
        # Applied as matmul in forward.

    def __call__(self, tokens: mx.array) -> mx.array:
        B, T = tokens.shape
        positions = mx.arange(T)[None, :]  # [1, T]
        x = self.token_embed(tokens) + self.pos_embed(positions)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        # Weight-tied projection back to vocab
        logits = x @ self.token_embed.weight.T
        return logits
