"""Validate or run checkpoint-dev; all three branches run in parallel by default."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from myeval.cli import app

import retool_coding_0812.myeval_plugin  # noqa: F401 - registers benchmark/executor
from retool_coding_0812.eval_resume import evaluation_run_lock, resume_mode
from retool_coding_0812.parallel import run_parallel
from retool_coding_0812.settings import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("c0", "c1", "c2"))
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.variant is None:
        commands = []
        labels = []
        for variant in ("c0", "c1", "c2"):
            config = ROOT / "configs/generated/checkpoint_dev" / f"{variant}.yaml"
            mode = "run" if args.validate_only else resume_mode(config, args.resume)
            if mode == "skip":
                print(f"[parallel] skipping completed {variant}", flush=True)
                continue
            command = [sys.executable, str(Path(__file__).resolve()), "--variant", variant]
            if args.validate_only:
                command.append("--validate-only")
            if mode == "resume":
                command.append("--resume")
            commands.append(command)
            labels.append(variant)
        run_parallel(commands, cwd=ROOT, labels=labels)
        return
    config = ROOT / "configs/generated/checkpoint_dev" / f"{args.variant}.yaml"
    mode = "run" if args.validate_only else resume_mode(config, args.resume)
    if mode == "skip":
        print(f"[parallel] skipping completed {args.variant}", flush=True)
        return
    command = "validate" if args.validate_only else "run"
    with evaluation_run_lock(config):
        sys.argv = ["myeval", command, str(config)]
        if mode == "resume" and not args.validate_only:
            sys.argv.append("--resume")
        app()


if __name__ == "__main__":
    main()
