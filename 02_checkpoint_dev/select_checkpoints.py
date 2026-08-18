"""Select checkpoints by cross-prompt average, worst, case-pass, then earlier step."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--myeval-runs", nargs=3, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, default=100)
    parser.add_argument("--required-steps", nargs="+", type=int, default=[20, 40, 60, 80, 100])
    return parser.parse_args()


def candidates(
    run_dir: Path,
    *,
    expected_tasks: int = 100,
    required_steps: set[int] | None = None,
) -> list[dict[str, object]]:
    """Extract only complete, task-paired, three-prompt checkpoint candidates."""

    summary = json.loads((run_dir / "summary" / "metrics.json").read_text(encoding="utf-8"))
    counts = dict(summary.get("counts") or {})
    steps = required_steps or {20, 40, 60, 80, 100}
    expected_cells_per_model = 3
    expected_completed = expected_tasks * len(steps) * expected_cells_per_model
    if counts.get("completed") != expected_completed or any(
        counts.get(status, 0) for status in ("failed", "pending", "running")
    ):
        raise ValueError(f"Dev run {run_dir} is incomplete or contains failures: {counts}")
    if not (summary.get("matrix_analysis") or {}).get("paired", False):
        raise ValueError(f"Dev run {run_dir} is not task-paired")
    config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    models = {str(item["id"]): dict(item) for item in config["models"]}
    by_model: dict[str, list[dict[str, object]]] = {}
    for cell in summary["matrix_analysis"]["cells"]:
        if int(cell.get("tasks", 0)) != expected_tasks:
            raise ValueError(
                f"Dev cell {cell.get('model_id')} does not contain {expected_tasks} tasks"
            )
        by_model.setdefault(str(cell["model_id"]), []).append(dict(cell))
    output = []
    for model_id, cells in sorted(by_model.items()):
        model = models[model_id]
        metadata = dict(model.get("metadata") or {})
        train_variant = str(metadata.get("train_system_prompt", "")).lower()
        step = int(metadata.get("checkpoint_step", 0))
        if step not in steps:
            continue
        prompts = {str(cell["test_prompt"]).lower() for cell in cells}
        expected_prompts = {"c0", "c1", "c2"}
        if prompts != expected_prompts or len(cells) != expected_cells_per_model:
            raise ValueError(f"Checkpoint {model_id} prompt coverage is incomplete: {prompts}")
        values = [float(cell["value"]) for cell in cells]
        pass_counts = [round(float(cell["value"]) * int(cell["tasks"])) for cell in cells]
        task_counts = [int(cell["tasks"]) for cell in cells]
        case_rates = [
            float((cell.get("metrics") or {}).get("case_pass_rate", 0.0))
            for cell in cells
        ]
        output.append(
            {
                "model_path": model.get("model_path"),
                "train_variant": train_variant,
                "seed": metadata.get("seed"),
                "step": step,
                "pass_at_1": sum(values) / len(values),
                "average": sum(values) / len(values),
                "worst": min(values),
                "case_pass_rate": sum(case_rates) / len(case_rates),
                # Keep exact count-based values for selection. The displayed float
                # averages can differ by one ULP for mathematically tied checkpoints.
                "pass_count": sum(pass_counts),
                "pass_total": sum(task_counts),
                "worst_pass_count": min(pass_counts),
                "worst_pass_total": task_counts[pass_counts.index(min(pass_counts))],
                "prompt_values": {
                    str(cell["test_prompt"]).lower(): float(cell["value"])
                    for cell in cells
                },
            }
        )
    return output


def main(args: argparse.Namespace) -> None:
    """Choose one checkpoint per branch using the preregistered lexicographic rule."""

    required_steps = set(getattr(args, "required_steps", [20, 40, 60, 80, 100]))
    expected_tasks = int(getattr(args, "expected_tasks", 100))
    grouped: dict[str, list[dict[str, object]]] = {}
    for run_dir in args.myeval_runs:
        for item in candidates(
            run_dir,
            expected_tasks=expected_tasks,
            required_steps=required_steps,
        ):
            variant = str(item["train_variant"])
            seed = item.get("seed")
            if seed != 42:
                raise ValueError(f"Only seed 42 is allowed, got {seed}")
            key = variant
            grouped.setdefault(key, []).append(item)
    selected: dict[str, object] = {}
    expected_groups = 3
    if len(grouped) != expected_groups:
        raise ValueError(f"Expected {expected_groups} checkpoint groups, got {len(grouped)}")
    for key, values in grouped.items():
        if len(values) != len(required_steps):
            raise ValueError(
                f"Expected {len(required_steps)} dev checkpoints for {key}, got {len(values)}"
            )
        selected[key] = max(
            values,
            key=lambda item: (
                Fraction(int(item["pass_count"]), int(item["pass_total"])),
                Fraction(
                    int(item["worst_pass_count"]), int(item["worst_pass_total"])
                ),
                float(item["case_pass_rate"]),
                -int(item["step"]),
            ),
        )
        for internal_key in (
            "pass_count",
            "pass_total",
            "worst_pass_count",
            "worst_pass_total",
        ):
            selected[key].pop(internal_key, None)
    selected["base"] = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main(parse_args())
