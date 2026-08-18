"""Seed-42 formal training entrypoint with no free-form hyperparameters."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from retool_coding_0812.resume import latest_checkpoint
from retool_coding_0812.settings import ROOT, branch_args


def _implementation():
    source = ROOT / "01_train" / "train_branch.py"
    spec = importlib.util.spec_from_file_location("retool_0812_train_branch", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load training implementation: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("c0", "c1", "c2"), required=True)
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume-checkpoint", type=Path)
    resume.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest local checkpoint for this branch.",
    )
    parser.add_argument(
        "--gate-only",
        action="store_true",
        help="Run all local reproducibility gates and exit before contacting PyTRIO.",
    )
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    resume_checkpoint = cli.resume_checkpoint
    if cli.resume:
        run_name = f"retool-coding-0812-{cli.variant}-seed42"
        resume_checkpoint = latest_checkpoint(
            ROOT / "artifacts/training/checkpoints",
            pattern=f"{run_name}-step-*.json",
            expected_run_name=run_name,
        )
        if resume_checkpoint is None:
            print(f"No checkpoint found for {run_name}; restarting safely from step 0")
    implementation = _implementation()
    implementation.main(
        branch_args(
            cli.variant,
            resume_checkpoint=resume_checkpoint,
            recover=cli.resume,
            gate_only=cli.gate_only,
        )
    )


if __name__ == "__main__":
    main()
