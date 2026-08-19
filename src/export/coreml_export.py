"""Export a quantized student model to CoreML (.mlpackage) for iPhone deployment.

The export pipeline:
  1. Load INT4 MLX weights, dequantize them back to FP32, and load them into a
     PyTorch mirror of StudentModel (a temporary numpy bridge — coremltools
     works with PyTorch or TF graphs, not MLX directly).
  2. Use ct.convert() with NeuralNetwork or ML Program backend.
  3. Set compute_units=CPU_AND_NE to activate the Neural Engine.

Note: coremltools requires either a TorchScript trace or an ONNX graph.
We use the ONNX path for maximum compatibility:
  StudentModel (MLX) → onnx via numpy weights → ct.convert(onnx_model)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import coremltools as ct
import mlx.core as mx
import numpy as np
import torch
import torch.nn as tnn
import torch.onnx


class _TorchAttention(tnn.Module):
    """Causal multi-head self-attention, mirroring src/model/student.py's MultiHeadAttention."""

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = tnn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = tnn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = tnn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = tnn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).reshape(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, H, D).transpose(1, 2)

        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        attn = (q @ k.transpose(-2, -1)) * self.scale + mask
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class _TorchFeedForward(tnn.Module):
    """SwiGLU-style gated FFN, mirroring src/model/student.py's FeedForward."""

    def __init__(self, hidden_dim: int, ff_dim: int) -> None:
        super().__init__()
        self.gate_proj = tnn.Linear(hidden_dim, ff_dim, bias=False)
        self.up_proj = tnn.Linear(hidden_dim, ff_dim, bias=False)
        self.down_proj = tnn.Linear(ff_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class _TorchTransformerBlock(tnn.Module):
    """Pre-norm block: LayerNorm → Attn + residual → LayerNorm → FFN + residual.

    Mirrors src/model/student.py's TransformerBlock.
    """

    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int) -> None:
        super().__init__()
        self.norm1 = tnn.LayerNorm(hidden_dim)
        self.attn = _TorchAttention(hidden_dim, num_heads)
        self.norm2 = tnn.LayerNorm(hidden_dim)
        self.ffn = _TorchFeedForward(hidden_dim, ff_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class _TorchStudentModel(tnn.Module):
    """PyTorch mirror of StudentModel (src/model/student.py) for ONNX tracing.

    Architecture and parameter names match the MLX original exactly (pre-norm
    blocks, causal attention, SwiGLU FFN, weight-tied lm_head), so a trained
    checkpoint's flattened parameter dict can be loaded directly via
    load_state_dict().
    """

    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int, num_heads: int, max_seq_len: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        ff_dim = hidden_dim * 4

        self.token_embed = tnn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = tnn.Embedding(max_seq_len, hidden_dim)
        self.blocks = tnn.ModuleList([
            _TorchTransformerBlock(hidden_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        self.norm = tnn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0)
        x = self.token_embed(tokens) + self.pos_embed(pos)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        # Weight-tied projection back to vocab
        logits = x @ self.token_embed.weight.T
        return logits


def _load_dequantized_state_dict(
    weights_path: str | Path,
    group_size: int = 64,
    bits: int = 4,
) -> dict[str, torch.Tensor]:
    """Load weights.npz (possibly INT4-quantized) and return an FP32 torch state dict.

    quantize_int4() (src/optimization/quantize.py) stores quantized Linear
    weights as three entries: "<name>" (packed 4-bit), "<name>_scales", and
    "<name>_biases". Every other entry (embeddings, LayerNorm params) is left
    as plain FP32. This reverses that: quantized entries are dequantized back
    to FP32 with mx.dequantize, and the rest are passed through.

    group_size and bits must match the values quantize_int4() was called
    with (its defaults are 64 and 4 respectively).
    """
    weights: dict[str, mx.array] = mx.load(str(weights_path))
    quantized_names = {name[: -len("_scales")] for name in weights if name.endswith("_scales")}

    state_dict: dict[str, torch.Tensor] = {}
    for name, tensor in weights.items():
        if name.endswith("_scales") or name.endswith("_biases"):
            continue
        if name in quantized_names:
            scales = weights[f"{name}_scales"]
            biases = weights.get(f"{name}_biases")
            fp32 = mx.dequantize(tensor, scales=scales, biases=biases, group_size=group_size, bits=bits)
        else:
            fp32 = tensor.astype(mx.float32)
        state_dict[name] = torch.from_numpy(np.array(fp32, dtype=np.float32))
    return state_dict


def export_coreml(
    int4_dir: str | Path,
    out_path: str | Path,
    compute_units: str = "CPU_AND_NE",
    max_seq_len: int = 512,
    group_size: int = 64,
    bits: int = 4,
) -> None:
    """Convert an INT4 MLX student model to a CoreML .mlpackage.

    Args:
        int4_dir: Directory with INT4 weights.npz and config.json.
        out_path: Destination path for the .mlpackage (must end with .mlpackage).
        compute_units: CoreML compute units string ("CPU_AND_NE", "ALL", "CPU_ONLY").
        max_seq_len: Sequence length for the fixed-shape ONNX trace.
        group_size: Quantization group size used by quantize_int4() on this model.
        bits: Quantization bit-width used by quantize_int4() on this model.
    """
    int4_dir = Path(int4_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((int4_dir / "config.json").read_text())
    vocab_size = cfg["vocab_size"]
    hidden_dim = cfg["hidden_dim"]
    num_layers = cfg["num_layers"]
    num_heads = cfg["num_heads"]
    cfg_max_seq_len = cfg.get("max_seq_len", max_seq_len)

    print("Building PyTorch model for ONNX trace …")
    torch_model = _TorchStudentModel(vocab_size, hidden_dim, num_layers, num_heads, cfg_max_seq_len)

    print(f"Loading trained weights from {int4_dir / 'weights.npz'} …")
    state_dict = _load_dequantized_state_dict(int4_dir / "weights.npz", group_size=group_size, bits=bits)
    torch_model.load_state_dict(state_dict)
    torch_model.eval()

    # Trace with a dummy input
    dummy = torch.zeros((1, max_seq_len), dtype=torch.long)
    onnx_path = str(out_path.with_suffix(".onnx"))
    print(f"Exporting ONNX → {onnx_path} …")
    torch.onnx.export(
        torch_model,
        dummy,
        onnx_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "logits": {0: "batch", 1: "seq"}},
        opset_version=17,
    )

    _cu_map = {
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "ALL": ct.ComputeUnit.ALL,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
    }
    cu = _cu_map.get(compute_units, ct.ComputeUnit.CPU_AND_NE)

    print(f"Converting to CoreML ({compute_units}) …")
    mlmodel = ct.convert(
        onnx_path,
        inputs=[ct.TensorType(name="input_ids", shape=(1, max_seq_len), dtype=np.int32)],
        compute_units=cu,
        minimum_deployment_target=ct.target.iOS17,
    )
    mlmodel.save(str(out_path))
    # Clean up intermediate ONNX
    os.remove(onnx_path)
    print(f"CoreML package saved → {out_path}")
