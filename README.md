# on-device-llm-optimizer

> **Shrink Phi-3 Mini from 3.8B to a ~236M on-device model: distill, INT4-quantize, and ship it to CoreML.**

[![CI](https://github.com/shaikn6/on-device-llm-optimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/shaikn6/on-device-llm-optimizer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)

**A pipeline to knowledge-distill a 3.8B teacher into a ~236M student, INT4-quantize it, and export it toward CoreML for the iPhone Neural Engine, on Apple Silicon (MLX).**

Phi-3 Mini 3.8B (frozen teacher) → custom 236M MLX student → group-wise INT4 →
CoreML `.mlpackage` (CPU + Neural Engine), with a Streamlit dashboard that compares all
four variants on size, speed, MMLU and perplexity.

The full Phi-3 → 236M distillation has not been run in this repo, so its speed and MMLU
figures are design targets ([Benchmark targets](#benchmark-targets)). What is measured: model
file sizes, a CoreML export of the full-size architecture that writes a `.mlpackage` and runs
on a Mac (not yet on an iPhone; [Verification status](#verification-status)), and a small
distillation smoke run where the distilled student beats a no-teacher control on held-out
perplexity (236.0 vs 350.2; [Measured smoke run](#measured-smoke-run)).

**Tech stack:** MLX (training + INT4 quantization), PyTorch TorchScript + coremltools (CoreML export bridge), Hugging Face datasets (Alpaca 52K), Streamlit + Plotly (dashboard), pytest.

## Architecture

```mermaid
flowchart TD
    subgraph Teacher
      T["Phi-3 Mini 3.8B<br/>microsoft/Phi-3-mini-4k-instruct<br/>BF16 as loaded, frozen"]
    end
    subgraph Distillation["Knowledge Distillation — src/distillation"]
      D["trainer.py loop<br/>Alpaca 52K · 10k steps · lr 3e-4"]
      L["losses.py · kd_loss<br/>L = α·KL(soft, T=4)·T² + (1−α)·CE<br/>α = 0.7"]
    end
    subgraph Student["Student — src/model"]
      S["StudentModel (MLX nn.Module)<br/>12 layers · 1024 dim · 8 heads<br/>SwiGLU FFN · weight-tied lm_head<br/>vocab 32064 → ~236M params"]
    end
    Q["INT4 quantize — src/optimization/quantize.py<br/>mx.quantize, group_size 64"]
    C["CoreML export — src/export/coreml_export.py<br/>MLX→Torch bridge → ct.convert<br/>ComputeUnit CPU_AND_NE · iOS17"]
    V["Streamlit dashboard — src/app<br/>size · tok/s · MMLU · perplexity"]

    T -->|soft labels| D
    D --> L
    L --> S
    S --> Q
    S --> C
    Q --> C
    Q --> V
    C --> V
```

Every box maps to a real module — the diagram is drawn from the actual source tree
and `configs/distill_config.yaml`, not an idealized design.

## How it works

1. **Teacher (frozen).** Phi-3 Mini 3.8B is used only to produce soft logits — it is
   never updated. `trainer.py` loads the pinned Hugging Face checkpoint as published
   (BF16, 7.64 GB of safetensors); the `teacher.quantize: int4` key in the config is not
   read by any code, so the teacher is not quantized.
2. **Distillation loss** (`src/distillation/losses.py`). `kd_loss` is a weighted sum of a
   soft KL term and a hard cross-entropy term:
   `L = α · KL(softmax(S/T) ‖ softmax(Teacher/T)) · T² + (1−α) · CE(S, hard_labels)`,
   with temperature `T = 4.0` and `α = 0.7` (config-driven). The `T²` factor restores
   gradient scale after temperature softening, as in Hinton et al. Note the KL direction:
   this implementation computes KL(student ‖ teacher) (reverse KL, mode-seeking), whereas
   Hinton et al. use KL(teacher ‖ student) (forward KL).
3. **Student** (`src/model/student.py`, `config.py`). A from-scratch MLX transformer:
   12 layers, hidden dim 1024, 8 heads, SwiGLU gated FFN (4× expansion), LayerNorm,
   learned positional embeddings, and an `lm_head` weight-tied to the token embedding.
   With `vocab_size = 32064` this yields **~236M parameters** (asserted in
   `tests/test_student_model.py`).
4. **Training loop** (`src/distillation/trainer.py`). Distills on the Alpaca 52K dataset
   for 10k steps (batch 8, lr 3e-4), checkpointing every 500 steps.
5. **INT4 quantization** (`src/optimization/quantize.py`). Each 2-D weight whose last
   dim is divisible by `group_size` (64) — the Linear layers and the token and position
   embeddings — is passed through `mx.quantize(bits=4)`, storing 4-bit packed weights
   plus per-group FP32 scales/biases.
6. **CoreML export** (`src/export/coreml_export.py`). MLX has no direct CoreML path, so the
   INT4 weights are dequantized into an equivalent Torch module, traced with TorchScript, and
   converted with `ct.convert(..., compute_units=CPU_AND_NE, minimum_deployment_target=iOS17)`
   to a `.mlpackage` targeting the Neural Engine. The package stores FP16 weights; it does not
   keep the 4-bit packing.
7. **Evaluation + dashboard** (`src/evaluation`, `src/app`). `perplexity.py` computes
   held-out perplexity; `mmlu_eval.py` scores MMLU; the Streamlit app renders the
   four-variant comparison.

## Benchmark targets

> **Tokens/sec and MMLU are design targets, not measured results.** This repo ships the
> full pipeline but no trained checkpoints (`checkpoints/` and `models/` are empty
> placeholders), so those two columns — and the matching values backing the dashboard
> charts in `src/app/streamlit_app.py` — are the project's goals for a completed
> distillation run on M-series hardware. Run the pipeline end-to-end to produce your own
> figures. The architecture facts (236M params, 12L × 1024d × 8h, INT4 group size 64) are
> real and test-verified.
>
> **Sizes are measured.** File size is fixed by the architecture, not by training, so it
> can be measured without a trained checkpoint (see the note below the table).

| Variant | Size (measured) | Tokens/sec (target) | MMLU (target) |
|---------|------|-----------|------|
| Teacher (Phi-3 Mini, as loaded) | 7.64 GB (BF16) | 25 tok/s | 68.8% |
| Student FP32 | 945.3 MB | 12 tok/s | ~57% |
| Student INT4 | 147.9 MB | 45 tok/s | ~54% |
| Student CoreML | 470.1 MB (FP16) | 68 tok/s (NE) | ~54% |

How the sizes were measured (MB = 10⁶ bytes):

- **Student FP32 / INT4** — build `StudentModel(StudentConfig())` (236,308,480 params,
  random init), save its parameters to `weights.npz` the way `trainer.py` does
  (`mx.savez` of `tree_flatten(model.parameters())`), then run the repo's own
  `quantize_int4(..., group_size=64)` on it and take the `weights.npz` file sizes:
  945,266,380 bytes FP32 and 147,942,532 bytes INT4. The INT4 file is 118.1 MB of packed
  4-bit weights + 29.5 MB of FP32 per-group scales/biases + 0.2 MB of unquantized
  vectors, so even the packed weights alone leave no room for the earlier 120 MB target
  at group size 64.
- **Teacher** — sum of the `.safetensors` files of `microsoft/Phi-3-mini-4k-instruct` at
  the revision pinned in `src/pins.py`, from the Hugging Face API. That is what
  `trainer.py` loads; it does not quantize the teacher.
- **CoreML** — `export_coreml()` run on that INT4 file (default 512-token trace, `CPU_AND_NE`)
  writes a `.mlpackage` of 470,079,295 bytes. CoreML stores the dequantized weights as FP16,
  so the package is about 3x the INT4 file; 4-bit compression inside CoreML (coremltools
  weight palettization or linear quantization) is not implemented.

## Measured smoke run

The speed and MMLU targets above are unmeasured. To show the distillation path actually learns — on the repo's own
`StudentModel`, `kd_loss` and trainer — `scripts/kd_smoke.py` runs a small, real KD job and writes
`results/kd_smoke.json`. It is a smoke run, **not** the Phi-3 → 236M target: a 16.8M student
(4L × 256d × 4h), the open Apache-2.0 teacher `SmolLM2-135M`, 500 steps on Tiny Shakespeare
(public-domain), contiguous train/val split so no val text is trained on. Held-out perplexity, lower is better:

| Model | Val perplexity |
|---|---|
| Teacher (SmolLM2-135M) | 34.8 |
| Student, random init | 76061 |
| Student, **KD** (α=0.7, T=4) | **236.0** |
| Student, CE-only control (α=0) | 350.2 |

The KD student (236) beats the no-teacher CE-only control (350) under the same budget, so the teacher
signal measurably helps. One seed (0) on Apple Silicon (MLX 0.32.4). Reproduce with
`python scripts/kd_smoke.py`; its defaults are the committed configuration
(`--steps 500 --batch-size 8 --seq-len 128 --lr 1e-3 --seed 0`, as recorded in `results/kd_smoke.json`).
It needs about 20 minutes on an M-series Mac: the committed run spent 495 s and 579 s training
the two students, plus teacher scoring.
`tests/test_kd_smoke.py` checks the pieces on tiny tensors with no downloads.

## Quickstart

```bash
pip install -e ".[dev]"
```

The `dev` extra adds the test and lint tools (`pytest`, `pytest-cov`, `flake8`); plain
`pip install -e .` is enough to run the pipeline. Verified in a clean Python 3.12 venv on
Apple Silicon (`python3 -m venv`, then `pip install -e ".[dev]"`): resolves and installs
cleanly, including `mlx`, `mlx-lm`, and `coremltools`.

CoreML export (`src/export/coreml_export.py`, step 3 below) also needs `torch`, which the
`dev` extra already includes. With a plain install, add it before running `scripts/export.py`:

```bash
pip install -e ".[export]"
```

## Verification status

What's confirmed by running the code, versus by reading it, on an Apple M2 Pro
(Python 3.12, clean venv, torch 2.15, coremltools 9.0):

- **Test suite** — `pytest tests/ -v --cov=src` (after `pip install -e ".[dev]"`) passes 75/75
  with 63% line coverage of `src/` (`TOTAL 609 224 63%`). (A bare `--cov` reports 92%, but that only counts modules the tests import,
  including the test files themselves; the trainer, dashboard and MMLU eval
  are not exercised by tests.) CI runs the real suite as a blocking gate: the MLX-free
  config/pins tests on `ubuntu-latest`, and the full suite on a `macos-latest` (Apple
  Silicon) runner where MLX installs. No coverage threshold is enforced. Covers the
  student model's forward shapes and ~236M parameter count, the distillation loss
  (`kd_loss`), INT4 quantize/dequantize round-tripping, padding-mask correctness in
  `batch_iter`, perplexity scoring, `StudentConfig`/revision-pin validation, and the CoreML
  export (`tests/test_coreml_export.py`: a tiny model is quantized, exported, loaded with
  coremltools, and on macOS its predicted logits must have the right shape and agree with the
  PyTorch mirror's argmax on at least 90% of positions).
- **CoreML export weight-loading and architecture match — verified numerically.**
  `_load_dequantized_state_dict` loads a saved `weights.npz` into the PyTorch
  mirror (`_TorchStudentModel`) in `src/export/coreml_export.py`. Feeding the
  same weights into both the MLX `StudentModel` and the PyTorch mirror and
  comparing logits on identical input gives a max absolute difference of
  `~1.5e-6` (float32 rounding noise) — the two implementations compute the
  same function.
- **`.mlpackage` write — verified on a Mac.** The export used to go through ONNX, but
  coremltools 6 and later have no ONNX frontend, so `ct.convert(onnx_path, ...)` failed with
  `ValueError: Unable to determine the type of the model`. It now traces the PyTorch mirror
  with TorchScript and converts that. On the full-size architecture (random weights) it writes
  a 470.1 MB package, and `predict()` on a Mac returns logits of the right shape. coremltools
  warns that torch versions newer than 2.7 are untested; this run used torch 2.15.
- **Not verified:** loading the package on an iPhone, Neural Engine placement, and speed.
- **Speed and MMLU numbers above are not measured** — see the note under Benchmark
  targets. The sizes in that table are measured.

## Usage

```bash
# 1. Download Alpaca 52K dataset
python scripts/download_data.py

# 2. Run distillation (not yet run end to end in this repo; expect several hours on an M-series Mac)
python scripts/train.py

# 3. Quantize + export to CoreML (needs torch, see Quickstart)
python scripts/export.py

# 4. Launch benchmark dashboard
streamlit run src/app/streamlit_app.py
```

## Tests

```bash
pytest tests/ -v
```

Covers the student model (param count, forward shapes), distillation losses, INT4
quantization, perplexity, CoreML export, and benchmark wiring.

The MLX-dependent tests need Apple Silicon. `tests/test_config.py` and `tests/test_pins.py`
import no MLX and reach no network, so they run anywhere with only `pytest` installed —
this is what the cross-platform CI job gates on:

```bash
pytest tests/test_config.py tests/test_pins.py -v
```

## Environment Variables

Copy `.env.example` to `.env` and fill in values if needed:

```bash
cp .env.example .env
```

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace token (optional, only for gated models) |
| `DISTILL_CONFIG` | Override config path (default: `configs/distill_config.yaml`) |

## Limitations

- **The headline pipeline is unmeasured.** No Phi-3 → 236M distillation has been run, so there
  is no trained student; tokens/sec and MMLU in the table are design targets, and the dashboard
  labels them as targets.
- **The only measured learning result is a smoke run**: a 16.8M student, a 135M teacher, 500 steps
  on Tiny Shakespeare, one seed. It shows the KD path learns, not that the 236M student will reach
  the MMLU target.
- **CoreML** is verified only as far as writing the package and running `predict()` on a Mac. It
  stores FP16 weights (470.1 MB, not 4-bit), and it has not been run on an iPhone or profiled on
  the Neural Engine.
- **Reverse KL.** `kd_loss` uses KL(student ‖ teacher), not the forward KL of Hinton et al.; the
  smoke run is the only evidence for this choice.
- **Apple Silicon only** for anything that imports MLX; CI runs the full suite on a macOS runner.

## Requirements

- macOS 13+ (Ventura) with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- 16 GB RAM minimum (32 GB recommended for full distillation run; estimates, since the full run has not been done)

## License

MIT

---
<sub>Part of [Nagizaaz Shaik's open-source work](https://github.com/shaikn6) · [all repositories](https://github.com/shaikn6?tab=repositories)</sub>
