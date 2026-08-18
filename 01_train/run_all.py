"""Run shared SFT once, then C0, C1, and C2 concurrently."""

from __future__ import annotations

import argparse
import subprocess
import sys

from retool_coding_0812.parallel import run_parallel
from retool_coding_0812.settings import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume every incomplete stage from its latest complete checkpoint.",
    )
    args = parser.parse_args()
    sft_manifest = ROOT / "artifacts/training/shared-sft/seed42/manifest.json"
    if args.gate_only:
        # Gate-only uses the copied, immutable e3 manifest as provenance and
        # never needs to contact PyTRIO to reproduce SFT first.
        pass
    elif args.resume or not sft_manifest.exists():
        command = [sys.executable, str(ROOT / "01_train/shared_sft.py")]
        if args.resume:
            command.append("--resume")
        subprocess.run(command, check=True, cwd=ROOT)
    commands = []
    variants = ("c0", "c1", "c2")
    for variant in variants:
        command = [sys.executable, str(ROOT / "01_train/run.py"), "--variant", variant]
        if args.gate_only:
            command.append("--gate-only")
        elif args.resume:
            command.append("--resume")
        commands.append(command)
    run_parallel(commands, cwd=ROOT, labels=variants)


if __name__ == "__main__":
    main()
