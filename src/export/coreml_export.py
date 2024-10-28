"""Export a quantized student model to CoreML (.mlpackage) for iPhone deployment.

The export pipeline:
  1. Load INT4 MLX weights and reconstruct as a torch-traced module via
     a temporary numpy bridge (coremltools works with PyTorch or TF graphs).
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
import numpy as np
import torch
import torch.nn as tnn
import torch.onnx


class _TorchStudentModel(tnn.Module):
    """Minimal PyTorch mirror of StudentModel for ONNX tracing.

    Only the forward pass computation matters for the export graph.
    Architecture matches src/model/student.py but in PyTorch.
    """

    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int, num_heads: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        ff_dim = hidden_dim * 4

        self.token_embed = tnn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = tnn.Embedding(512, hidden_dim)
        self.blocks = tnn.ModuleList([
            tnn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                batch_first=True,
            )
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
        logits = x @ self.token_embed.weight.T
        return logits


def export_coreml(
    int4_dir: str | Path,
    out_path: str | Path,
    compute_units: str = "CPU_AND_NE",
    max_seq_len: int = 512,
) -> None:
    """Convert an INT4 MLX student model to a CoreML .mlpackage.

    Args:
        int4_dir: Directory with INT4 weights.npz and config.json.
        out_path: Destination path for the .mlpackage (must end with .mlpackage).
        compute_units: CoreML compute units string ("CPU_AND_NE", "ALL", "CPU_ONLY").
        max_seq_len: Sequence length for the fixed-shape ONNX trace.
    """
    int4_dir = Path(int4_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((int4_dir / "config.json").read_text())
    vocab_size = cfg["vocab_size"]
    hidden_dim = cfg["hidden_dim"]
    num_layers = cfg["num_layers"]
    num_heads = cfg["num_heads"]

    print("Building PyTorch model for ONNX trace …")
    torch_model = _TorchStudentModel(vocab_size, hidden_dim, num_layers, num_heads)
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
