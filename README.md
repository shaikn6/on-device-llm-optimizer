> **Private Repository** — Source code available on request for verified employers.
> Contact: shaikn6@udayton.edu

# On-Device LLM Optimizer

Knowledge distillation pipeline: Phi-3 Mini 3.8B → 236M student model on Apple Silicon (MLX).
Quantizes to INT4. Exports to CoreML (.mlpackage) targeting the iPhone Neural Engine.
Benchmarks all four variants in a Streamlit dashboard.

## Architecture

```
Teacher: Phi-3 Mini 3.8B (INT4, frozen)
    │  soft labels (T=4)
    ▼
Knowledge Distillation (Alpaca 52K, 10K steps)
    Loss = 0.7 × KL_soft + 0.3 × CE_hard
    │
    ▼
Student: Custom 236M Transformer (12 layers, 1024 dim, 8 heads)
    │              │
INT4 Quant     CoreML Export (.mlpackage, CPU+NE)
    │              │
    └──────────────┘
           ▼
Streamlit Dashboard (size · speed · MMLU · perplexity)
```

## Setup

```bash
pip install -e .
```

## Usage

```bash
# 1. Download Alpaca 52K dataset
python scripts/download_data.py

# 2. Run distillation (~6-10 hours on M-series Mac)
python scripts/train.py

# 3. Quantize + export to CoreML
python scripts/export.py

# 4. Launch benchmark dashboard
streamlit run src/app/streamlit_app.py
```

## Benchmark Targets

| Variant | Size | Tokens/sec | MMLU |
|---------|------|-----------|------|
| Teacher (Phi-3 INT4) | 2.2 GB | 25 tok/s | 68.8% |
| Student FP32 | 910 MB | 12 tok/s | ~57% |
| Student INT4 | 120 MB | 45 tok/s | ~54% |
| Student CoreML | 120 MB | 68 tok/s (NE) | ~54% |

## Tests

```bash
pytest tests/ -v
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

## Requirements

- macOS 13+ (Ventura) with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- 16 GB RAM minimum (32 GB recommended for full distillation run)
