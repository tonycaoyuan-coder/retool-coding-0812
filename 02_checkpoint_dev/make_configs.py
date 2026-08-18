"""Generate the three frozen checkpoint-dev MyEval configs."""

from __future__ import annotations

import argparse
from pathlib import Path

from retool_coding_0812.eval_config import checkpoint_dev_config, write_yaml
from retool_coding_0812.settings import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        type=Path,
        default=ROOT / "artifacts/training/checkpoints/models.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "configs/generated/checkpoint_dev",
    )
    args = parser.parse_args()
    for variant in ("c0", "c1", "c2"):
        destination = args.output_dir / f"{variant}.yaml"
        write_yaml(destination, checkpoint_dev_config(variant, args.models))
        print(destination)


if __name__ == "__main__":
    main()
