"""Merge three prompt-sharded runs into the complete 12-cell final report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import statistics
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--myeval-runs", nargs=3, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def merge_summaries(run_dirs: list[Path]) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    deltas: list[dict[str, Any]] = []
    mcnemar: list[dict[str, Any]] = []
    total_completed = 0
    observed_prompts: set[str] = set()
    paired_sample_sets: list[set[str]] = []
    for run_dir in run_dirs:
        summary_path = run_dir / "summary" / "metrics.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        counts = dict(summary.get("counts") or {})
        if counts.get("completed") != 800 or any(
            counts.get(status, 0) for status in ("failed", "pending", "running")
        ):
            raise ValueError(f"Final shard {run_dir} is incomplete or contains failures: {counts}")
        matrix = dict(summary.get("matrix_analysis") or {})
        shard_cells = list(matrix.get("cells") or [])
        shard_deltas = list(matrix.get("deltas_vs_base") or [])
        shard_mcnemar = list(matrix.get("mcnemar_vs_base") or [])
        if len(shard_cells) != 4 or len(shard_deltas) != 3 or len(shard_mcnemar) != 3:
            raise ValueError(f"Final shard {run_dir} does not contain one complete prompt column")
        prompts = {str(cell["test_prompt"]).lower() for cell in shard_cells}
        if len(prompts) != 1:
            raise ValueError(f"Final shard {run_dir} spans multiple test prompts: {sorted(prompts)}")
        prompt = next(iter(prompts))
        if prompt in observed_prompts:
            raise ValueError(f"Duplicate final test-prompt shard: {prompt}")
        if not matrix.get("paired", False):
            raise ValueError(f"Final shard {run_dir} is not task-paired")
        state_path = run_dir / "state.sqlite"
        with sqlite3.connect(state_path) as connection:
            rows = connection.execute(
                "SELECT model_id, sample_id FROM tasks WHERE status = 'completed'"
            ).fetchall()
        sample_sets: dict[str, set[str]] = {}
        for model_id, sample_id in rows:
            sample_sets.setdefault(str(model_id), set()).add(str(sample_id))
        if set(sample_sets) != {"base", "c0", "c1", "c2"}:
            raise ValueError(f"Final shard {run_dir} has incomplete model pairing")
        unique_sets = {frozenset(values) for values in sample_sets.values()}
        if len(unique_sets) != 1:
            raise ValueError(f"Final shard {run_dir} has mismatched sample IDs")
        paired_sample_sets.append(set(next(iter(unique_sets))))
        observed_prompts.add(prompt)
        total_completed += int(counts["completed"])
        cells.extend(shard_cells)
        deltas.extend(shard_deltas)
        mcnemar.extend(shard_mcnemar)

    if observed_prompts != {"c0", "c1", "c2"}:
        raise ValueError(f"Final test-prompt shards are incomplete: {sorted(observed_prompts)}")
    if len({frozenset(values) for values in paired_sample_sets}) != 1:
        raise ValueError("Final prompt shards do not contain identical task IDs")
    by_train: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        train = str(cell["train_prompt"]).lower()
        if train != "base":
            by_train.setdefault(train, []).append(cell)
    robustness = []
    for train in ("c0", "c1", "c2"):
        train_cells = by_train.get(train, [])
        if len(train_cells) != 3:
            raise ValueError(f"Final train row {train} is incomplete")
        values = [float(cell["value"]) for cell in train_cells]
        diagonal = next(
            float(cell["value"])
            for cell in train_cells
            if str(cell["test_prompt"]).lower() == train
        )
        off_diagonal = [
            float(cell["value"])
            for cell in train_cells
            if str(cell["test_prompt"]).lower() != train
        ]
        robustness.append(
            {
                "benchmark_id": train_cells[0]["benchmark_id"],
                "train_prompt": train,
                "checkpoint_step": int(train_cells[0]["checkpoint_step"]),
                "average": statistics.fmean(values),
                "worst": min(values),
                "overfit_gap": diagonal - statistics.fmean(off_diagonal),
            }
        )
    return {
        "counts": {"completed": total_completed, "failed": 0, "pending": 0, "running": 0},
        "matrix_analysis": {
            "cells": cells,
            "paired": True,
            "pairing_errors": {},
            "robustness": robustness,
            "deltas_vs_base": deltas,
            "mcnemar_vs_base": mcnemar,
        },
    }


def main(args: argparse.Namespace) -> None:
    summary = merge_summaries(list(args.myeval_runs))
    matrix = dict(summary["matrix_analysis"])
    cells = list(matrix["cells"])
    if len(cells) != 12:
        raise ValueError(f"Final report requires 12 cells, got {len(cells)}")
    if not matrix.get("paired", False):
        raise ValueError(f"Final cells are not task-paired: {matrix.get('pairing_errors')}")
    counts = dict(summary["counts"])
    if counts.get("completed") != 2400 or any(
        counts.get(status, 0) for status in ("failed", "pending", "running")
    ):
        raise ValueError(f"Final run is incomplete or contains failures: {counts}")
    if any(int(cell.get("tasks", 0)) != 200 for cell in cells):
        raise ValueError("Every final cell must contain exactly the same 200 tasks")
    for key, expected_count in (
        ("robustness", 3),
        ("deltas_vs_base", 9),
        ("mcnemar_vs_base", 9),
    ):
        if len(matrix.get(key) or []) != expected_count:
            raise ValueError(
                f"Final matrix requires {expected_count} {key} rows, "
                f"got {len(matrix.get(key) or [])}"
            )
    expected = {
        (train, test.upper())
        for train in ("base", "c0", "c1", "c2")
        for test in ("c0", "c1", "c2")
    }
    actual = {(str(cell["train_prompt"]).lower(), str(cell["test_prompt"])) for cell in cells}
    if actual != expected:
        raise ValueError(f"Missing/unexpected cells: {sorted(expected ^ actual)}")
    by_key = {
        (str(cell["train_prompt"]).lower(), str(cell["test_prompt"]).lower()): cell
        for cell in cells
    }
    lines = [
        "# ReTool-Coding-0812 Seed-42 Final Matrix",
        "",
        "> Seed 42 only; 200 temporally held-out LCB-v6 tasks. This is not an official contamination-free LiveCodeBench score.",
        "",
        "| Train \\ Test | C0 | C1 | C2 | Average | Worst | OverfitGap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    robustness = {
        str(item["train_prompt"]).lower(): item
        for item in matrix.get("robustness", [])
    }
    for train in ("c0", "c1", "c2", "base"):
        values = [float(by_key[(train, test)]["value"]) for test in ("c0", "c1", "c2")]
        if train == "base":
            average, worst, gap = sum(values) / 3, min(values), "-"
        else:
            row = robustness[train]
            average, worst = float(row["average"]), float(row["worst"])
            gap = f"{float(row['overfit_gap']):.3f}"
        lines.append(
            f"| {train.upper()} | {values[0]:.3f} | {values[1]:.3f} | {values[2]:.3f} | "
            f"{average:.3f} | {worst:.3f} | {gap} |"
        )
    lines.extend(
        [
            "",
            "## Paired deltas vs Base",
            "",
            "| Train | Test | Delta | CI95 | Wins/Ties/Losses |",
            "|---|---|---:|---|---|",
        ]
    )
    for item in matrix.get("deltas_vs_base", []):
        lines.append(
            f"| {str(item['train_prompt']).upper()} | {str(item['test_prompt']).upper()} | "
            f"{float(item['delta']):.3f} | [{float(item['ci_low']):.3f}, {float(item['ci_high']):.3f}] | "
            f"{item['wins']}/{item['ties']}/{item['losses']} |"
        )
    lines.extend(
        [
            "",
            "## Exact McNemar vs Base",
            "",
            "| Train | Test | Trained-only | Base-only | Discordant | Exact p |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for item in matrix.get("mcnemar_vs_base", []):
        lines.append(
            f"| {str(item['train_prompt']).upper()} | {str(item['test_prompt']).upper()} | "
            f"{int(item['trained_only'])} | {int(item['base_only'])} | "
            f"{int(item['discordant'])} | {float(item['p_value']):.4f} |"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main(parse_args())
