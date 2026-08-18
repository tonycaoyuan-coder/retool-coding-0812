"""Recompute evaluation metrics and compare L16K/T24K with L10K/T20K."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
import re
from statistics import median
from typing import Any, Iterable

import numpy as np


MODELS = ("base", "c0", "c1", "c2")
MODEL_LABELS = {"base": "Base", "c0": "C0@40", "c1": "C1@100", "c2": "C2@100"}
PROMPTS = ("C0", "C1", "C2")
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260816
BINARY_METRICS = ("pass_at_1", "format_valid", "cap_hit")
COMPARISON_METRICS = (
    "pass_at_1",
    "case_pass_rate",
    "format_valid",
    "cap_hit",
    "first_turn_cap",
    "completion_tokens",
    "first_turn_tokens",
    "unsubmitted_tokens",
    "line_repeat_ratio",
    "ngram8_repeat_ratio",
    "revision_markers_per_1k",
    "latency_seconds",
)
REVISION_PATTERN = re.compile(
    r"\b(?:but\s+wait|wait|actually|let(?:'|’)s\s+try|try\s+(?:a\s+)?different|"
    r"different\s+approach|doesn(?:'|’)t\s+help|not\s+right|rethink|start\s+over|"
    r"go\s+back)\b",
    flags=re.IGNORECASE,
)
TOKEN_PATTERN = re.compile(
    r"[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|==|!=|<=|>=|//|<<|>>|\*\*|\S"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-runs", nargs=3, type=Path, required=True)
    parser.add_argument("--baseline-runs", nargs=3, type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return float(np.mean(items)) if items else math.nan


def bootstrap_ci(values: Iterable[float], seed_offset: int = 0) -> list[float]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return [math.nan, math.nan]
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    draws = rng.choice(array, size=(BOOTSTRAP_SAMPLES, array.size), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def exact_mcnemar_p(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def normalize_line(line: str) -> str | None:
    value = re.sub(r"^#+\s*", "", line.strip())
    value = re.sub(r"\s+", " ", value).strip().lower()
    if len(value) < 12:
        return None
    if value.startswith(("```", "<tool_call", "</tool_call", "<function", "</function")):
        return None
    if value.startswith(("<parameter", "</parameter")):
        return None
    return value


def repeated_line_ratio(turn_texts: list[str]) -> float:
    repeated = total = 0
    for text in turn_texts:
        lines = [value for line in text.splitlines() if (value := normalize_line(line))]
        counts = Counter(lines)
        repeated += sum(count - 1 for count in counts.values() if count > 1)
        total += len(lines)
    return repeated / total if total else 0.0


def repeated_ngram_ratio(turn_texts: list[str], n: int = 8) -> float:
    repeated = total = 0
    for text in turn_texts:
        tokens = [token.lower() for token in TOKEN_PATTERN.findall(text)]
        ngrams = [tuple(tokens[index : index + n]) for index in range(max(0, len(tokens) - n + 1))]
        counts = Counter(ngrams)
        repeated += sum(count - 1 for count in counts.values() if count > 1)
        total += len(ngrams)
    return repeated / total if total else 0.0


def trajectory_row(path: Path, experiment: str) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        trajectory = json.load(stream)
    turns = list(trajectory.get("turns") or [])
    texts = [str(turn.get("text") or "") for turn in turns]
    counts = [int(turn.get("completion_token_count") or 0) for turn in turns]
    completion_tokens = sum(counts)
    final_code = trajectory.get("final_code")
    format_valid = float(final_code is not None)
    unsubmitted = sum(counts[:-1]) if final_code is not None else completion_tokens
    judge = dict(trajectory.get("judge_result") or {})
    total = int(judge.get("total") or 0)
    public_total = int(judge.get("public_total") or 0)
    private_total = int(judge.get("private_total") or 0)
    status = str(judge.get("status") or "unknown")
    attempts = int(trajectory.get("tool_call_attempts") or 0)
    valid_calls = int(trajectory.get("valid_tool_calls") or 0)
    tool_calls = int(trajectory.get("tool_calls") or 0)
    first_prompt_tokens = len(turns[0].get("prompt_tokens") or []) if turns else 0
    trajectory_tokens = (
        len(turns[-1].get("prompt_tokens") or []) + len(turns[-1].get("completion_tokens") or [])
        if turns
        else 0
    )
    revision_markers = sum(len(REVISION_PATTERN.findall(text)) for text in texts)
    example = dict(trajectory.get("example") or {})
    return {
        "experiment": experiment,
        "sample_id": str(example.get("instance_id") or ""),
        "title": str(example.get("question_title") or ""),
        "platform": str(example.get("platform") or "unknown"),
        "difficulty": str(example.get("difficulty") or "unknown"),
        "contest_date": example.get("contest_date"),
        "model": str(trajectory["metadata"]["model_id"]).lower(),
        "test_prompt": str(trajectory["metadata"]["system_prompt_id"]).upper(),
        "status": status,
        "pass_at_1": float(bool(judge.get("resolved"))),
        "case_pass_rate": int(judge.get("passed") or 0) / max(total, 1),
        "public_pass_rate": int(judge.get("public_passed") or 0) / max(public_total, 1),
        "private_pass_rate": int(judge.get("private_passed") or 0) / max(private_total, 1),
        "format_valid": format_valid,
        "cap_hit": float(bool(trajectory.get("hit_token_limit"))),
        "first_turn_cap": float(bool(turns and turns[0].get("hit_token_limit"))),
        "compile_error": float(status == "compile_error"),
        "runtime_error": float(status == "runtime_error"),
        "time_limit": float(status == "time_limit"),
        "tool_call_attempts": attempts,
        "valid_tool_calls": valid_calls,
        "tool_calls": tool_calls,
        "tool_attempted": float(attempts > 0),
        "tool_used": float(tool_calls > 0),
        "turns": len(turns),
        "prompt_tokens": first_prompt_tokens,
        "completion_tokens": completion_tokens,
        "trajectory_tokens": trajectory_tokens,
        "first_turn_tokens": counts[0] if counts else 0,
        "unsubmitted_tokens": unsubmitted,
        "final_turn_tokens": counts[-1] if final_code is not None and counts else 0,
        "line_repeat_ratio": repeated_line_ratio(texts),
        "ngram8_repeat_ratio": repeated_ngram_ratio(texts),
        "revision_markers": revision_markers,
        "revision_markers_per_1k": 1000.0 * revision_markers / max(completion_tokens, 1),
        "execution_seconds": float(judge.get("execution_seconds") or 0.0),
        "latency_seconds": float(trajectory.get("duration_seconds") or 0.0),
        "artifact_path": str(path.resolve()),
    }


def load_rows(run_dirs: list[Path], experiment: str) -> list[dict[str, Any]]:
    rows = [
        trajectory_row(path, experiment)
        for run_dir in run_dirs
        for path in sorted((run_dir / "artifacts/trajectories").glob("*.json.gz"))
    ]
    expected = len(MODELS) * len(PROMPTS) * 200
    if len(rows) != expected:
        raise ValueError(f"{experiment}: expected {expected} trajectories, found {len(rows)}")
    keys = {(row["model"], row["test_prompt"], row["sample_id"]) for row in rows}
    if len(keys) != expected:
        raise ValueError(f"{experiment}: trajectories are not uniquely paired")
    if {row["model"] for row in rows} != set(MODELS):
        raise ValueError(f"{experiment}: unexpected model set")
    if {row["test_prompt"] for row in rows} != set(PROMPTS):
        raise ValueError(f"{experiment}: unexpected prompt set")
    return rows


MEAN_FIELDS = (
    "pass_at_1",
    "case_pass_rate",
    "public_pass_rate",
    "private_pass_rate",
    "format_valid",
    "cap_hit",
    "first_turn_cap",
    "compile_error",
    "runtime_error",
    "time_limit",
    "tool_attempted",
    "tool_used",
    "tool_calls",
    "valid_tool_calls",
    "turns",
    "prompt_tokens",
    "completion_tokens",
    "trajectory_tokens",
    "first_turn_tokens",
    "unsubmitted_tokens",
    "line_repeat_ratio",
    "ngram8_repeat_ratio",
    "revision_markers_per_1k",
    "execution_seconds",
    "latency_seconds",
)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {f"mean_{field}": mean(float(row[field]) for row in rows) for field in MEAN_FIELDS}
    valid = [row for row in rows if row["format_valid"] > 0.5]
    tool = [row for row in rows if row["tool_calls"] > 0]
    no_tool = [row for row in rows if row["tool_calls"] == 0]
    cap = [row for row in rows if row["cap_hit"] > 0.5]
    failed = [row for row in rows if row["pass_at_1"] < 0.5]
    cap_hit_failures = [row for row in failed if row["cap_hit"] > 0.5]
    attempts = sum(int(row["tool_call_attempts"]) for row in rows)
    result.update(
        {
            "n": len(rows),
            "median_completion_tokens": float(median(row["completion_tokens"] for row in rows)),
            "final_turn_share": sum(row["final_turn_tokens"] for row in rows)
            / max(sum(row["completion_tokens"] for row in rows), 1),
            "pass_given_valid": mean(row["pass_at_1"] for row in valid),
            "pass_given_tool_use": mean(row["pass_at_1"] for row in tool),
            "pass_given_no_tool": mean(row["pass_at_1"] for row in no_tool),
            "valid_tool_attempt_rate": sum(row["valid_tool_calls"] for row in rows)
            / max(attempts, 1),
            "cap_only_line_repeat_ratio": mean(row["line_repeat_ratio"] for row in cap),
            "invalid_given_cap": mean(1.0 - row["format_valid"] for row in cap),
            "pass_given_cap": mean(row["pass_at_1"] for row in cap),
            "pass_given_no_cap": mean(
                row["pass_at_1"] for row in rows if row["cap_hit"] < 0.5
            ),
            "failure_count": len(failed),
            "cap_hit_failure_count": len(cap_hit_failures),
            "cap_hit_share_among_failures": len(cap_hit_failures) / max(len(failed), 1),
            "statuses": dict(sorted(Counter(row["status"] for row in rows).items())),
        }
    )
    return result


def grouped_summaries(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[key]) for key in keys)].append(row)
    return {"|".join(key): summarize(values) for key, values in sorted(grouped.items())}


def paired_effects(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    *,
    condition: str,
) -> list[dict[str, Any]]:
    left = {(row["model"], row["test_prompt"], row["sample_id"]): row for row in left_rows}
    right = {(row["model"], row["test_prompt"], row["sample_id"]): row for row in right_rows}
    if set(left) != set(right):
        raise ValueError(f"Pairing failed for {condition}")
    effects = []
    for model in MODELS:
        for prompt in PROMPTS:
            keys = sorted(key for key in left if key[0] == model and key[1] == prompt)
            metrics: dict[str, Any] = {}
            for offset, metric in enumerate(COMPARISON_METRICS):
                deltas = [float(right[key][metric]) - float(left[key][metric]) for key in keys]
                item: dict[str, Any] = {
                    "delta": mean(deltas),
                    "ci95": bootstrap_ci(deltas, seed_offset=offset),
                }
                if metric in BINARY_METRICS:
                    wins = sum(left[key][metric] < 0.5 and right[key][metric] > 0.5 for key in keys)
                    losses = sum(left[key][metric] > 0.5 and right[key][metric] < 0.5 for key in keys)
                    item.update(
                        {
                            "wins": wins,
                            "ties": len(keys) - wins - losses,
                            "losses": losses,
                            "mcnemar_p": exact_mcnemar_p(wins, losses),
                        }
                    )
                metrics[metric] = item
            baseline_caps = [key for key in keys if left[key]["cap_hit"] > 0.5]
            added_tokens = [right[key]["completion_tokens"] - left[key]["completion_tokens"] for key in keys]
            effects.append(
                {
                    "condition": condition,
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "test_prompt": prompt,
                    "n": len(keys),
                    "metrics": metrics,
                    "baseline_cap_n": len(baseline_caps),
                    "baseline_cap_to_valid": sum(right[key]["format_valid"] > 0.5 for key in baseline_caps),
                    "baseline_cap_to_pass": sum(right[key]["pass_at_1"] > 0.5 for key in baseline_caps),
                    "baseline_cap_still_cap": sum(right[key]["cap_hit"] > 0.5 for key in baseline_caps),
                    "no_outcome_gain_with_more_tokens": sum(
                        added > 0
                        and right[key]["pass_at_1"] <= left[key]["pass_at_1"]
                        and right[key]["format_valid"] <= left[key]["format_valid"]
                        for key, added in zip(keys, added_tokens, strict=True)
                    ),
                }
            )
    return effects


def within_experiment_effects(rows: list[dict[str, Any]]) -> dict[str, Any]:
    table = {(row["model"], row["test_prompt"], row["sample_id"]): row for row in rows}
    samples = sorted({row["sample_id"] for row in rows})
    prompt_pairs = []
    for model in MODELS:
        for left_prompt, right_prompt in (("C0", "C1"), ("C0", "C2"), ("C1", "C2")):
            for metric in ("pass_at_1", "cap_hit", "format_valid", "completion_tokens"):
                deltas = [
                    table[(model, right_prompt, sample)][metric]
                    - table[(model, left_prompt, sample)][metric]
                    for sample in samples
                ]
                prompt_pairs.append(
                    {
                        "model": model,
                        "contrast": f"{right_prompt}-{left_prompt}",
                        "metric": metric,
                        "delta": mean(deltas),
                        "ci95": bootstrap_ci(deltas),
                    }
                )
    vs_base = []
    for model in ("c0", "c1", "c2"):
        for prompt in PROMPTS:
            for metric in ("pass_at_1", "cap_hit", "format_valid", "completion_tokens"):
                deltas = [
                    table[(model, prompt, sample)][metric] - table[("base", prompt, sample)][metric]
                    for sample in samples
                ]
                vs_base.append(
                    {
                        "model": model,
                        "test_prompt": prompt,
                        "metric": metric,
                        "delta": mean(deltas),
                        "ci95": bootstrap_ci(deltas),
                    }
                )
    return {"prompt_pairs": prompt_pairs, "models_vs_base": vs_base}


def representative_cases(
    baseline: list[dict[str, Any]], new: list[dict[str, Any]], limit: int = 20
) -> dict[str, Any]:
    left = {(row["model"], row["test_prompt"], row["sample_id"]): row for row in baseline}
    right = {(row["model"], row["test_prompt"], row["sample_id"]): row for row in new}
    rescued = []
    wasted = []
    for key in sorted(left):
        old, current = left[key], right[key]
        record = {
            "model": key[0],
            "test_prompt": key[1],
            "sample_id": key[2],
            "title": current["title"],
            "baseline_tokens": old["completion_tokens"],
            "new_tokens": current["completion_tokens"],
            "baseline_artifact": old["artifact_path"],
            "new_artifact": current["artifact_path"],
        }
        if old["cap_hit"] > 0.5 and old["pass_at_1"] < 0.5 and current["pass_at_1"] > 0.5:
            rescued.append(record)
        if (
            current["completion_tokens"] > old["completion_tokens"]
            and current["pass_at_1"] <= old["pass_at_1"]
            and current["format_valid"] <= old["format_valid"]
        ):
            wasted.append(record | {"added_tokens": current["completion_tokens"] - old["completion_tokens"]})
    wasted.sort(key=lambda item: item["added_tokens"], reverse=True)
    return {"cap_to_pass": rescued[:limit], "largest_no-outcome-gain": wasted[:limit]}


def write_row_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [key for key in rows[0] if key != "contest_date"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def write_cell_csv(path: Path, cells: dict[str, Any]) -> None:
    rows = []
    for key, summary in sorted(cells.items()):
        model, prompt = key.split("|")
        rows.append({"model": model, "test_prompt": prompt, **{k: v for k, v in summary.items() if not isinstance(v, dict)}})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt_delta(item: dict[str, Any], scale: float = 1.0, digits: int = 3) -> str:
    low, high = item["ci95"]
    return f"{item['delta'] * scale:+.{digits}f} [{low * scale:+.{digits}f}, {high * scale:+.{digits}f}]"


def write_report(path: Path, analysis: dict[str, Any]) -> None:
    effects = analysis["baseline_vs_new"]
    lines = [
        "# ReTool-Coding-0812 L16K/24K Length Ablation",
        "",
        "> Post-hoc inference-length ablation; seed 42; greedy decoding; the same 200 held-out tasks and frozen checkpoints as the original L10K/20K final evaluation.",
        "",
        "## Paired cell comparison: L16K/24K minus L10K/20K",
        "",
        "| Model | Test | Δ pass@1 | Δ format-valid | Δ cap-hit | Δ completion | Baseline cap → pass | Baseline cap still cap |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in effects:
        metrics = item["metrics"]
        lines.append(
            f"| {item['model_label']} | {item['test_prompt']} | "
            f"{fmt_delta(metrics['pass_at_1'], scale=100, digits=1)}pp | "
            f"{fmt_delta(metrics['format_valid'], scale=100, digits=1)}pp | "
            f"{fmt_delta(metrics['cap_hit'], scale=100, digits=1)}pp | "
            f"{fmt_delta(metrics['completion_tokens'], digits=1)} | "
            f"{item['baseline_cap_to_pass']}/{item['baseline_cap_n']} | "
            f"{item['baseline_cap_still_cap']}/{item['baseline_cap_n']} |"
        )
    baseline_cap_n = sum(item["baseline_cap_n"] for item in effects)
    cap_to_pass = sum(item["baseline_cap_to_pass"] for item in effects)
    cap_to_valid = sum(item["baseline_cap_to_valid"] for item in effects)
    still_cap = sum(item["baseline_cap_still_cap"] for item in effects)
    no_gain = sum(item["no_outcome_gain_with_more_tokens"] for item in effects)
    lines.extend(
        [
            "",
            "## Outcome accounting",
            "",
            f"- Original cap-hit trajectories: `{baseline_cap_n}`.",
            f"- Converted to a valid final submission: `{cap_to_valid}/{baseline_cap_n}`.",
            f"- Converted to a fully correct submission: `{cap_to_pass}/{baseline_cap_n}`.",
            f"- Still hit a token cap: `{still_cap}/{baseline_cap_n}`.",
            f"- Used more completion tokens without improving pass or format outcome: `{no_gain}/2400`.",
            "",
            "The paired deltas isolate inference-budget behavior for these fixed checkpoints. They do not establish a new training effect.",
            "",
            "## Truncation share among failed tasks",
            "",
            "A failed task has `pass@1 = 0`; a truncated task has `cap_hit = 1` on any assistant turn. The percentage denominator is failed tasks in that cell, not all 200 tasks.",
            "",
            "| Model | Test | L10K/20K truncated / failed | L16K/24K truncated / failed |",
            "|---|---|---:|---:|",
        ]
    )
    for model in MODELS:
        for prompt in PROMPTS:
            key = f"{model}|{prompt}"
            old = analysis["baseline_cells"][key]
            new = analysis["new_cells"][key]
            lines.append(
                f"| {MODEL_LABELS[model]} | {prompt} | "
                f"{old['cap_hit_failure_count']}/{old['failure_count']} "
                f"({old['cap_hit_share_among_failures']:.2%}) | "
                f"{new['cap_hit_failure_count']}/{new['failure_count']} "
                f"({new['cap_hit_share_among_failures']:.2%}) |"
            )
    old_failed = sum(cell["failure_count"] for cell in analysis["baseline_cells"].values())
    old_cap_failed = sum(
        cell["cap_hit_failure_count"] for cell in analysis["baseline_cells"].values()
    )
    new_failed = sum(cell["failure_count"] for cell in analysis["new_cells"].values())
    new_cap_failed = sum(
        cell["cap_hit_failure_count"] for cell in analysis["new_cells"].values()
    )
    lines.append(
        f"| **Overall** | — | **{old_cap_failed}/{old_failed} "
        f"({old_cap_failed / old_failed:.2%})** | **{new_cap_failed}/{new_failed} "
        f"({new_cap_failed / new_failed:.2%})** |"
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = load_rows(list(args.baseline_runs), "l10k-t20k")
    new = load_rows(list(args.new_runs), "l16k-t24k")
    new_cells = grouped_summaries(new, ("model", "test_prompt"))
    analysis = {
        "provenance": {
            "baseline_runs": [str(path.resolve()) for path in args.baseline_runs],
            "new_runs": [str(path.resolve()) for path in args.new_runs],
            "models": list(MODELS),
            "prompts": list(PROMPTS),
            "tasks_per_cell": 200,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "baseline_cells": grouped_summaries(baseline, ("model", "test_prompt")),
        "new_cells": new_cells,
        "new_model_marginals": grouped_summaries(new, ("model",)),
        "new_prompt_marginals": grouped_summaries(new, ("test_prompt",)),
        "new_difficulty": grouped_summaries(new, ("difficulty", "model")),
        "new_platform": grouped_summaries(new, ("platform", "model")),
        "new_within_experiment_effects": within_experiment_effects(new),
        "baseline_vs_new": paired_effects(baseline, new, condition="l16k-t24k-minus-l10k-t20k"),
        "representative_cases": representative_cases(baseline, new),
    }
    write_row_csv(args.output_dir / "l16k-t24k-row-metrics.csv", new)
    write_cell_csv(args.output_dir / "l16k-t24k-cell-metrics.csv", new_cells)
    (args.output_dir / "l16k-t24k-analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    write_report(args.output_dir / "l16k-t24k-vs-l10k-t20k.md", analysis)
    print(args.output_dir)


if __name__ == "__main__":
    main(parse_args())
