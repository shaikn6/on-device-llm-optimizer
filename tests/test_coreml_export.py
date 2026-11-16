"""Tests for TorchScript-based CoreML export."""
import json
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ct = pytest.importorskip("coremltools")

import mlx.core as mx
import mlx.utils as mx_utils

from src.export.coreml_export import (
    _TorchStudentModel,
    _load_dequantized_state_dict,
    export_coreml,
)
from src.model.config import StudentConfig
from src.model.student import StudentModel
from src.optimization.quantize import quantize_int4


def test_coreml_export(tmp_path):
    """Export quantized weights and compare CoreML predictions to the mirror."""
    mx.random.seed(0)
    cfg = StudentConfig(num_layers=1, hidden_dim=64, num_heads=2, vocab_size=128, max_seq_len=16)
    model = StudentModel(cfg)
    fp32_dir = tmp_path / "fp32"
    fp32_dir.mkdir()
    weights = dict(mx_utils.tree_flatten(model.parameters()))
    mx.savez(str(fp32_dir / "weights.npz"), **weights)
    (fp32_dir / "config.json").write_text(json.dumps(cfg.__dict__))

    group_size = 64
    int4_dir = tmp_path / "int4"
    quantize_int4(fp32_dir, int4_dir, group_size=group_size)
    out_path = tmp_path / "student.mlpackage"
    export_coreml(int4_dir, out_path, max_seq_len=8, group_size=group_size)
    assert out_path.is_dir()

    mlmodel = ct.models.MLModel(str(out_path))
    if sys.platform == "darwin":
        input_ids = np.arange(8, dtype=np.int32).reshape(1, 8)
        predictions = mlmodel.predict({"input_ids": input_ids})
        output_name = mlmodel.get_spec().description.output[0].name
        logits = predictions[output_name]
        assert logits.shape == (1, 8, cfg.vocab_size)

        torch_model = _TorchStudentModel(
            cfg.vocab_size, cfg.hidden_dim, cfg.num_layers, cfg.num_heads, cfg.max_seq_len,
        )
        torch_model.load_state_dict(
            _load_dequantized_state_dict(int4_dir / "weights.npz", group_size=group_size)
        )
        torch_model.eval()
        with torch.no_grad():
            expected = torch_model(torch.from_numpy(input_ids).long()).numpy()
        assert np.mean(logits.argmax(axis=-1) == expected.argmax(axis=-1)) >= 0.9
