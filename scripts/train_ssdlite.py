"""Fine-tune SSDLite with a reviewed real-scene manifest and a separate run directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Prepared dataset.json manifest")
    parser.add_argument("--output", type=Path, required=True, help="New run directory, or the same run with --resume")
    parser.add_argument("--initial-weights", type=Path, required=True, help="Existing trained SSDLite checkpoint")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=0.0003)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-background-samples", type=int, default=32)
    parser.add_argument("--confidence-threshold", type=float, default=0.29)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    options = parser.parse_args(argv)
    from src.detection.training import TrainingConfig, run_training

    report = run_training(TrainingConfig(**vars(options)))
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
