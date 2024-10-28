"""Entry point: quantize the FP32 student then export to CoreML.

Usage:
    python scripts/export.py [--config configs/distill_config.yaml]
"""
import argparse
import os
from pathlib import Path

import yaml

from src.optimization.quantize import quantize_int4
from src.export.coreml_export import export_coreml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quantize + export the student model")
    p.add_argument(
        "--config",
        default=os.getenv("DISTILL_CONFIG", "configs/distill_config.yaml"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    q_cfg = cfg["quantization"]
    ex_cfg = cfg["export"]

    print("=== Step 1: INT4 Quantization ===")
    quantize_int4(
        fp32_dir="models/student_fp32",
        out_dir="models/student_int4",
        group_size=q_cfg["group_size"],
    )

    print("\n=== Step 2: CoreML Export ===")
    export_coreml(
        int4_dir="models/student_int4",
        out_path="models/student.mlpackage",
        compute_units=ex_cfg["compute_units"],
    )

    print("\nExport complete.")


if __name__ == "__main__":
    main()
