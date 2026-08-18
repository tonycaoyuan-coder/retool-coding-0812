"""Collect an exact per-branch set of sampler checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_REQUIRED_STEPS = {20, 40, 60, 80, 100}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--required-steps", nargs="+", type=int, default=sorted(DEFAULT_REQUIRED_STEPS)
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    required_steps = set(args.required_steps)
    if not required_steps or min(required_steps) < 1:
        raise ValueError("Required checkpoint steps must be positive")
    models = {}
    seen = {variant: set() for variant in ("c0", "c1", "c2")}
    for path in sorted(args.checkpoint_dir.glob("*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        variant = str(item.get("prompt_variant", "")).lower()
        step = int(item.get("step", 0))
        if variant not in seen or step not in required_steps:
            continue
        if step in seen[variant]:
            raise ValueError(f"Duplicate checkpoint for {variant} step {step}: {path}")
        if not item.get("state_path") or not item.get("sampler_weights_path"):
            raise ValueError(f"Checkpoint record lacks state or sampler weights: {path}")
        models[f"{variant}-step-{step}"] = {
            "model_path": item["sampler_weights_path"],
            "train_variant": variant,
            "step": step,
            "seed": int(item.get("seed", -1)),
        }
        if models[f"{variant}-step-{step}"]["seed"] != 42:
            raise ValueError(f"Only seed 42 checkpoints are allowed: {path}")
        seen[variant].add(step)
    missing = {
        variant: sorted(required_steps - steps)
        for variant, steps in seen.items()
        if steps != required_steps
    }
    if missing:
        raise ValueError(f"Checkpoint set is incomplete: {missing}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(models, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(models, indent=2))


if __name__ == "__main__":
    main(parse_args())
