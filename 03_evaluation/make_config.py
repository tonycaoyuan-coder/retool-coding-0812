"""Generate three prompt-sharded final evaluation configs."""

from __future__ import annotations

import argparse
from pathlib import Path

from retool_coding_0812.eval_config import final_config, write_yaml
from retool_coding_0812.settings import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        type=Path,
        default=ROOT / "artifacts/checkpoint_dev/selected-models.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--experiment-tag")
    parser.add_argument("--max-assistant-tokens", type=int)
    parser.add_argument("--max-trajectory-tokens", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--tasks", type=int)
    parser.add_argument("--timeout-seconds", type=int)
    args = parser.parse_args()
    output_dir = args.output_dir or (
        ROOT / f"configs/generated/evaluation-{args.experiment_tag}"
        if args.experiment_tag
        else ROOT / "configs/generated/evaluation"
    )
    for variant in ("c0", "c1", "c2"):
        destination = output_dir / f"{variant}.yaml"
        write_yaml(
            destination,
            final_config(
                args.models,
                variant,
                max_assistant_tokens=args.max_assistant_tokens,
                max_trajectory_tokens=args.max_trajectory_tokens,
                experiment_tag=args.experiment_tag,
                output_root=args.output_root,
                tasks=args.tasks,
                execution_timeout_seconds=args.timeout_seconds,
            ),
        )
        print(destination)


if __name__ == "__main__":
    main()
