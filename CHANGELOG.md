# Changelog

## [Unreleased]
### Added
- CPU-only unit tests (`tests/test_config.py`, `tests/test_pins.py`) that import no MLX
  and reach no network, so they run on any platform

### Changed
- CI now gates on the real test suite: the MLX-free tests run on `ubuntu-latest` and the
  full suite runs on a `macos-latest` (Apple Silicon) runner, both blocking, replacing the
  previous non-blocking test step
- `torch` is declared: a new `export` extra, also included in `dev`

### Fixed
- CoreML export no longer goes through ONNX, which coremltools 6+ cannot read (`ValueError:
  Unable to determine the type of the model`); it traces the PyTorch mirror with TorchScript.
  `tests/test_coreml_export.py` exports a tiny model and checks its predictions; the full-size
  package (FP16 weights) measures 470.1 MB

## [0.1.0] - 2026-06-17
### Added
- MLX-based knowledge distillation pipeline: a frozen Phi-3 Mini 3.8B teacher (loaded as BF16)
  guides training of a from-scratch ~236M-parameter MLX student model
- INT4 weight quantization of the trained student (`src/optimization/quantize.py`)
- CoreML export path (MLX → PyTorch mirror → `coremltools`) producing an `.mlpackage`
  targeting CPU + Neural Engine
- MMLU and perplexity evaluation utilities to compare student variants
- Streamlit + Plotly dashboard comparing model size, speed, MMLU, and perplexity
  across teacher/student/quantized/CoreML variants
