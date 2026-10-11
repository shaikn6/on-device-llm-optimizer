"""Pre-fetch the Alpaca 52K dataset to the HuggingFace cache.

Usage:
    python scripts/download_data.py

This script has no arguments — it simply warms the HF dataset cache.
Subsequent calls to load_alpaca() will be instant.
"""
from datasets import load_dataset

# Same pin as src/pins.py; this script runs standalone, without the package on the path.
ALPACA_REVISION = "dce01c9b08f87459cf36a430d809084718273017"

if __name__ == "__main__":
    print("Downloading tatsu-lab/alpaca …")
    ds = load_dataset("tatsu-lab/alpaca", split="train", revision=ALPACA_REVISION)
    print(f"  Downloaded {len(ds):,} examples.")
    print("Done. Dataset is cached. Run scripts/train.py to start distillation.")
