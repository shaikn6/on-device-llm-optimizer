# Changelog

## [Unreleased]

## [1.0.0] - 2026-06-17
### Added
- MLX-based knowledge distillation pipeline: a frozen Phi-3 Mini 3.8B (INT4) teacher
  guides training of a from-scratch ~236M-parameter MLX student model
- INT4 weight quantization of the trained student (`src/optimization/quantize.py`)
- CoreML export path (MLX → PyTorch mirror → `coremltools`) producing an `.mlpackage`
  targeting CPU + Neural Engine
- MMLU and perplexity evaluation utilities to compare student variants
- Streamlit + Plotly dashboard comparing model size, speed, MMLU, and perplexity
  across teacher/student/quantized/CoreML variants
