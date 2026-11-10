"""Prepare all selected real detection images with immutable split provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.detection_manifest import prepare_detection_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits-source", type=Path, help="Directory containing authoritative train/val/test.txt manifests")
    parser.add_argument("--allow-background", action="store_true", help="Treat missing labels as explicitly authorized background images")
    parser.add_argument("--extra-train", type=Path, help="Visual review JSON containing training-only backgrounds and/or quarantine decisions")
    args = parser.parse_args(argv)
    try:
        manifest = prepare_detection_manifest(args.source, args.output, splits_source=args.splits_source,
                                              allow_background=args.allow_background, extra_train=args.extra_train)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Detection data preparation failed: {error}\n")
    print(json.dumps({"manifest": str((args.output / "dataset.json").resolve()),
                      "fingerprint": manifest["fingerprint"], "counts": manifest["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
