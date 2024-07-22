"""Entry point: run knowledge distillation.

Usage:
    python scripts/train.py [--config configs/distill_config.yaml]

Environment:
    HF_TOKEN   — Optional HuggingFace token for gated models.
"""
import argparse
import os
from pathlib import Path

import yaml

from src.distillation.trainer import DistillationTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distill Phi-3 Mini → student model")
    p.add_argument(
        "--config",
        default=os.getenv("DISTILL_CONFIG", "configs/distill_config.yaml"),
        help="Path to distill_config.yaml",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    cfg = yaml.safe_load(config_path.read_text())
    trainer = DistillationTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
