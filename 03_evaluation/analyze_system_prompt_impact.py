"""Reproducible analysis of train/test system-prompt effects in the 0812 run.

The local score, prediction, trajectory, checkpoint-dev, and training artifacts
are authoritative.  This script performs no network or remote-model calls.
"""

from __future__ import annotations

from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DATASET = ROOT.parent / "07-retool-lcb-mini" / "datasets" / "formal-v6" / "test.jsonl.gz"
OUTPUT = ARTIFACTS / "analysis/system-prompt-impact-analysis.json"
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260813

MODELS = ("base", "c0", "c1", "c2")
TRAINED_MODELS = ("c0", "c1", "c2")
POSTHOC_MODELS = ("c0-step100", "shared-sft-only")
ALL_EVAL_MODELS = ("base", "shared-sft-only", "c0", "c0-step100", "c1", "c2")
PROMPTS = ("C0", "C1", "C2")
EVAL_METRICS = (
    "pass_at_1",
    "case_pass_rate",
    "public_pass_rate",
    "private_pass_rate",
    "format_valid_rate",
    "token_cap_hit_rate",
    "compile_error_rate",
    "runtime_error_rate",
    "time_limit_rate",
    "tool_use_rate",
    "mean_tool_calls",
    "tool_call_valid_rate",
    "mean_turns",
    "mean_trajectory_tokens",
    "mean_execution_seconds",
)
EXTRA_METRICS = ("prompt_tokens", "completion_tokens", "latency_seconds")


def mean(rows: Iterable[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return float(np.mean(values)) if values else math.nan


def bootstrap_ci(values: Iterable[float]) -> list[float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return [math.nan, math.nan]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.choice(array, size=(BOOTSTRAP_SAMPLES, array.size), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def exact_mcnemar_p(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def holm_adjust(items: list[dict[str, Any]], key: str = "p_value") -> None:
    ordered = sorted(enumerate(items), key=lambda pair: float(pair[1][key]))
    running = 0.0
    total = len(items)
    for rank, (index, item) in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * float(item[key]))
        running = max(running, adjusted)
        items[index]["p_holm"] = running


def load_dataset_metadata() -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    with gzip.open(DATASET, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            metadata[str(row["instance_id"])] = {
                "title": str(row.get("question_title") or row["instance_id"]),
                "platform": str(row.get("platform") or "unknown"),
                "difficulty": str(row.get("difficulty") or "unknown"),
                "contest_date": row.get("contest_date"),
            }
    return metadata


def load_evaluation_rows(
    artifact_glob: str,
    expected_models: tuple[str, ...],
) -> list[dict[str, Any]]:
    dataset = load_dataset_metadata()
    rows: list[dict[str, Any]] = []
    for score_path in sorted(ARTIFACTS.glob(artifact_glob)):
        run_dir = score_path.parents[1]
        prediction_path = run_dir / "predictions" / score_path.name
        predictions = {
            row["task_key"]: row
            for row in (json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines())
            if row.get("status") == "completed"
        }
        for line in score_path.read_text(encoding="utf-8").splitlines():
            score_row = json.loads(line)
            if score_row.get("status") != "completed":
                continue
            prediction = predictions[score_row["task_key"]]
            score = dict(score_row["score"])
            metrics = dict(score["metrics"])
            result = dict(prediction["result"])
            details = dict(score.get("details") or {})
            judge = dict(details.get("judge_result") or result.get("metadata", {}).get("judge_result") or {})
            artifact_path = run_dir / str(details["artifact_path"])
            with gzip.open(artifact_path, "rt", encoding="utf-8") as stream:
                trajectory = json.load(stream)
            sample_id = str(score_row["sample_id"])
            row = {
                "sample_id": sample_id,
                "model": str(score_row["model_id"]).lower(),
                "test_prompt": str(score_row["condition"]["system_prompt_id"]).upper(),
                "status": str(judge.get("status", "unknown")),
                "tool_call_attempts": int(trajectory.get("tool_call_attempts", 0)),
                "valid_tool_calls": int(trajectory.get("valid_tool_calls", 0)),
                "tool_calls": int(trajectory.get("tool_calls", 0)),
                "hit_token_limit_raw": float(bool(trajectory.get("hit_token_limit", False))),
                "prompt_tokens": float(result.get("prompt_tokens", 0)),
                "completion_tokens": float(result.get("completion_tokens", 0)),
                "latency_seconds": float(result.get("latency_seconds", 0.0)),
                **dataset[sample_id],
            }
            row.update({key: float(metrics[key]) for key in EVAL_METRICS})
            rows.append(row)
    expected = len(expected_models) * len(PROMPTS) * len(dataset)
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} evaluation rows, found {len(rows)}")
    keys = {(row["sample_id"], row["model"], row["test_prompt"]) for row in rows}
    if len(keys) != expected:
        raise ValueError("Evaluation rows are not uniquely paired")
    found_models = {row["model"] for row in rows}
    if found_models != set(expected_models):
        raise ValueError(f"Expected models {expected_models}, found {sorted(found_models)}")
    return rows


def load_final_rows() -> list[dict[str, Any]]:
    return load_evaluation_rows("evaluation/*/*/scores/*.jsonl", MODELS)


def load_posthoc_rows() -> list[dict[str, Any]]:
    return load_evaluation_rows("evaluation-posthoc/*/*/scores/*.jsonl", POSTHOC_MODELS)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {key: mean(rows, key) for key in (*EVAL_METRICS, *EXTRA_METRICS)}
    failed_rows = [row for row in rows if row["pass_at_1"] < 0.5]
    cap_hit_failures = [row for row in failed_rows if row["token_cap_hit_rate"] > 0.5]
    format_count = sum(row["format_valid_rate"] for row in rows)
    summary["pass_given_valid_format"] = sum(row["pass_at_1"] for row in rows) / max(format_count, 1.0)
    summary["failure_count"] = len(failed_rows)
    summary["cap_hit_failure_count"] = len(cap_hit_failures)
    summary["cap_hit_share_among_failures"] = len(cap_hit_failures) / max(len(failed_rows), 1)
    attempts = sum(row["tool_call_attempts"] for row in rows)
    summary["attempted_tool_call_rate"] = sum(row["tool_call_attempts"] > 0 for row in rows) / len(rows)
    summary["valid_tool_attempt_rate"] = sum(row["valid_tool_calls"] for row in rows) / max(attempts, 1)
    tool_rows = [row for row in rows if row["tool_calls"] > 0]
    direct_rows = [row for row in rows if row["tool_calls"] == 0]
    summary["pass_given_tool_use"] = mean(tool_rows, "pass_at_1")
    summary["pass_given_no_tool"] = mean(direct_rows, "pass_at_1")
    summary["n"] = len(rows)
    statuses = defaultdict(int)
    for row in rows:
        statuses[row["status"]] += 1
    summary["statuses"] = dict(sorted(statuses.items()))
    return summary


def group_summaries(
    rows: list[dict[str, Any]],
    key_fn: Callable[[dict[str, Any]], Any],
) -> dict[str, dict[str, Any]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    return {str(key): summarize_rows(value) for key, value in sorted(groups.items(), key=lambda item: str(item[0]))}


def paired_values(
    rows: list[dict[str, Any]],
    *,
    group_key: str,
    condition_key: str,
    condition_a: str,
    condition_b: str,
    metric: str,
    filters: dict[str, str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    filters = filters or {}
    table: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if any(str(row[key]) != value for key, value in filters.items()):
            continue
        condition = str(row[condition_key])
        if condition in (condition_a, condition_b):
            table[str(row[group_key])][condition] = float(row[metric])
    complete = [values for values in table.values() if condition_a in values and condition_b in values]
    if not complete:
        raise ValueError(f"No complete pairs for {condition_a} vs {condition_b}")
    a = np.asarray([values[condition_a] for values in complete], dtype=np.float64)
    b = np.asarray([values[condition_b] for values in complete], dtype=np.float64)
    return a, b


def prompt_pair_effects(
    rows: list[dict[str, Any]],
    models: tuple[str, ...] = MODELS,
) -> list[dict[str, Any]]:
    effects: list[dict[str, Any]] = []
    for model in models:
        for prompt_a, prompt_b in (("C0", "C1"), ("C0", "C2"), ("C1", "C2")):
            a, b = paired_values(
                rows,
                group_key="sample_id",
                condition_key="test_prompt",
                condition_a=prompt_a,
                condition_b=prompt_b,
                metric="pass_at_1",
                filters={"model": model},
            )
            delta = b - a
            wins = int(np.sum((b == 1) & (a == 0)))
            losses = int(np.sum((a == 1) & (b == 0)))
            effects.append(
                {
                    "model": model,
                    "contrast": f"{prompt_b}-{prompt_a}",
                    "delta": float(delta.mean()),
                    "ci95": bootstrap_ci(delta),
                    "wins": wins,
                    "ties": int(len(delta) - wins - losses),
                    "losses": losses,
                    "p_value": exact_mcnemar_p(wins, losses),
                }
            )
    holm_adjust(effects)
    return effects


def model_pair_effects(
    rows: list[dict[str, Any]],
    pairs: tuple[tuple[str, str], ...],
) -> list[dict[str, Any]]:
    """Paired pass@1 effects for model B minus model A, separately by prompt."""

    effects: list[dict[str, Any]] = []
    table = {(row["sample_id"], row["model"], row["test_prompt"]): row for row in rows}
    samples = sorted({row["sample_id"] for row in rows})
    for model_a, model_b in pairs:
        for prompt in PROMPTS:
            differences = []
            wins = losses = 0
            for sample_id in samples:
                value_a = table[(sample_id, model_a, prompt)]["pass_at_1"]
                value_b = table[(sample_id, model_b, prompt)]["pass_at_1"]
                differences.append(value_b - value_a)
                wins += int(value_b == 1 and value_a == 0)
                losses += int(value_b == 0 and value_a == 1)
            effects.append(
                {
                    "contrast": f"{model_b}-{model_a}",
                    "test_prompt": prompt,
                    "delta": float(np.mean(differences)),
                    "ci95": bootstrap_ci(differences),
                    "wins": wins,
                    "ties": len(differences) - wins - losses,
                    "losses": losses,
                    "p_value": exact_mcnemar_p(wins, losses),
                }
            )
    return effects


def per_sample_condition_means(
    rows: list[dict[str, Any]],
    *,
    condition_key: str,
    metric: str,
    filters: dict[str, set[str]] | None = None,
) -> dict[str, dict[str, float]]:
    filters = filters or {}
    raw: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if any(str(row[key]) not in allowed for key, allowed in filters.items()):
            continue
        raw[str(row["sample_id"])][str(row[condition_key])].append(float(row[metric]))
    return {
        sample: {condition: float(np.mean(values)) for condition, values in conditions.items()}
        for sample, conditions in raw.items()
    }


def marginal_contrasts(
    rows: list[dict[str, Any]],
    *,
    condition_key: str,
    conditions: tuple[str, ...],
    metrics: tuple[str, ...],
    filters: dict[str, set[str]] | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for metric in metrics:
        per_sample = per_sample_condition_means(
            rows, condition_key=condition_key, metric=metric, filters=filters
        )
        for index, condition_a in enumerate(conditions):
            for condition_b in conditions[index + 1 :]:
                deltas = [
                    values[condition_b] - values[condition_a]
                    for values in per_sample.values()
                    if condition_a in values and condition_b in values
                ]
                results.append(
                    {
                        "metric": metric,
                        "contrast": f"{condition_b}-{condition_a}",
                        "delta": float(np.mean(deltas)),
                        "ci95": bootstrap_ci(deltas),
                        "n_tasks": len(deltas),
                    }
                )
    return results


def primary_vs_base(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    effects: list[dict[str, Any]] = []
    table = {(row["sample_id"], row["model"], row["test_prompt"]): row for row in rows}
    for model in TRAINED_MODELS:
        for prompt in PROMPTS:
            differences = []
            wins = losses = 0
            for sample_id in sorted({row["sample_id"] for row in rows}):
                trained = table[(sample_id, model, prompt)]["pass_at_1"]
                base = table[(sample_id, "base", prompt)]["pass_at_1"]
                differences.append(trained - base)
                wins += int(trained == 1 and base == 0)
                losses += int(trained == 0 and base == 1)
            effects.append(
                {
                    "model": model,
                    "test_prompt": prompt,
                    "delta": float(np.mean(differences)),
                    "ci95": bootstrap_ci(differences),
                    "wins": wins,
                    "ties": len(differences) - wins - losses,
                    "losses": losses,
                    "p_value": exact_mcnemar_p(wins, losses),
                }
            )
    holm_adjust(effects)
    return effects


def matching_effect(rows: list[dict[str, Any]]) -> dict[str, Any]:
    table = {(row["sample_id"], row["model"], row["test_prompt"]): row for row in rows}
    deltas = []
    for sample_id in sorted({row["sample_id"] for row in rows}):
        diagonal = np.mean(
            [table[(sample_id, model, model.upper())]["pass_at_1"] for model in TRAINED_MODELS]
        )
        off_diagonal = np.mean(
            [
                table[(sample_id, model, prompt)]["pass_at_1"]
                for model in TRAINED_MODELS
                for prompt in PROMPTS
                if prompt != model.upper()
            ]
        )
        deltas.append(float(diagonal - off_diagonal))
    return {
        "diagonal_mean": float(
            np.mean(
                [
                    row["pass_at_1"]
                    for row in rows
                    if row["model"] in TRAINED_MODELS and row["test_prompt"] == row["model"].upper()
                ]
            )
        ),
        "off_diagonal_mean": float(
            np.mean(
                [
                    row["pass_at_1"]
                    for row in rows
                    if row["model"] in TRAINED_MODELS and row["test_prompt"] != row["model"].upper()
                ]
            )
        ),
        "premium": float(np.mean(deltas)),
        "ci95": bootstrap_ci(deltas),
    }


def cap_format_association(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cap = np.asarray([row["token_cap_hit_rate"] > 0.5 for row in rows])
    invalid = np.asarray([row["format_valid_rate"] < 0.5 for row in rows])
    passed = np.asarray([row["pass_at_1"] > 0.5 for row in rows])
    return {
        "n": len(rows),
        "cap_hits": int(cap.sum()),
        "invalid_format": int(invalid.sum()),
        "cap_and_invalid": int(np.sum(cap & invalid)),
        "invalid_given_cap": float(np.mean(invalid[cap])) if cap.any() else math.nan,
        "cap_given_invalid": float(np.mean(cap[invalid])) if invalid.any() else math.nan,
        "pass_given_cap": float(np.mean(passed[cap])) if cap.any() else math.nan,
        "pass_given_no_cap": float(np.mean(passed[~cap])) if (~cap).any() else math.nan,
    }


def stratified_pass(
    rows: list[dict[str, Any]],
    field: str,
    models: tuple[str, ...] = MODELS,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for value in sorted({str(row[field]) for row in rows}):
        subset = [row for row in rows if str(row[field]) == value]
        n_tasks = len({row["sample_id"] for row in subset})
        for model in models:
            model_rows = [row for row in subset if row["model"] == model]
            result.append(
                {
                    field: value,
                    "model": model,
                    "n_tasks": n_tasks,
                    "pass_at_1": mean(model_rows, "pass_at_1"),
                    "case_pass_rate": mean(model_rows, "case_pass_rate"),
                    "format_valid_rate": mean(model_rows, "format_valid_rate"),
                    "token_cap_hit_rate": mean(model_rows, "token_cap_hit_rate"),
                }
            )
    return result


def variance_spans(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (*EVAL_METRICS, *EXTRA_METRICS)
    result = []
    for metric in metrics:
        model_means = {model: mean([row for row in rows if row["model"] == model], metric) for model in MODELS}
        prompt_means = {
            prompt: mean([row for row in rows if row["test_prompt"] == prompt], metric)
            for prompt in PROMPTS
        }
        result.append(
            {
                "metric": metric,
                "train_model_span": max(model_means.values()) - min(model_means.values()),
                "test_prompt_span": max(prompt_means.values()) - min(prompt_means.values()),
                "model_means": model_means,
                "prompt_means": prompt_means,
            }
        )
    return result


def representative_cases(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    table = {(row["sample_id"], row["model"], row["test_prompt"]): row for row in rows}
    samples = sorted({row["sample_id"] for row in rows})

    def record(sample_id: str) -> dict[str, Any]:
        first = table[(sample_id, "base", "C0")]
        return {
            "sample_id": sample_id,
            "title": first["title"],
            "platform": first["platform"],
            "difficulty": first["difficulty"],
            "cells": {
                f"{model.upper()}×{prompt}": {
                    "pass": int(table[(sample_id, model, prompt)]["pass_at_1"]),
                    "status": table[(sample_id, model, prompt)]["status"],
                    "case_pass_rate": table[(sample_id, model, prompt)]["case_pass_rate"],
                    "cap": int(table[(sample_id, model, prompt)]["token_cap_hit_rate"]),
                    "tokens": int(table[(sample_id, model, prompt)]["mean_trajectory_tokens"]),
                }
                for model in MODELS
                for prompt in PROMPTS
            },
        }

    categories: dict[str, list[str]] = {
        "base_fails_all_trained_pass_all": [],
        "c0_fails_all_c1_c2_pass_all": [],
        "c2_c0_passes_c1_c2_fail": [],
        "c1_c0_fails_c1_or_c2_passes": [],
        "all_models_cap_on_any_prompt": [],
    }
    for sample_id in samples:
        base = [table[(sample_id, "base", prompt)]["pass_at_1"] for prompt in PROMPTS]
        trained = [
            table[(sample_id, model, prompt)]["pass_at_1"]
            for model in TRAINED_MODELS
            for prompt in PROMPTS
        ]
        c0 = [table[(sample_id, "c0", prompt)]["pass_at_1"] for prompt in PROMPTS]
        c1 = [table[(sample_id, "c1", prompt)]["pass_at_1"] for prompt in PROMPTS]
        c2 = [table[(sample_id, "c2", prompt)]["pass_at_1"] for prompt in PROMPTS]
        if not any(base) and all(trained):
            categories["base_fails_all_trained_pass_all"].append(sample_id)
        if not any(c0) and all(c1 + c2):
            categories["c0_fails_all_c1_c2_pass_all"].append(sample_id)
        if c2[0] == 1 and c2[1] == 0 and c2[2] == 0:
            categories["c2_c0_passes_c1_c2_fail"].append(sample_id)
        if c1[0] == 0 and (c1[1] == 1 or c1[2] == 1):
            categories["c1_c0_fails_c1_or_c2_passes"].append(sample_id)
        if all(
            any(table[(sample_id, model, prompt)]["token_cap_hit_rate"] for prompt in PROMPTS)
            for model in MODELS
        ):
            categories["all_models_cap_on_any_prompt"].append(sample_id)
    return {
        category: [record(sample_id) for sample_id in sample_ids[:5]]
        for category, sample_ids in categories.items()
    } | {"category_counts": {key: len(value) for key, value in categories.items()}}


def load_checkpoint_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(ARTIFACTS.glob("checkpoint_dev/*/*/summary/metrics.json")):
        branch = path.parents[2].name
        payload = json.loads(path.read_text(encoding="utf-8"))
        for metric_row in payload["metrics"]:
            match = re.search(r"step-(\d+)$", str(metric_row["model_id"]))
            if not match:
                raise ValueError(f"Cannot parse checkpoint step from {metric_row['model_id']}")
            rows.append(
                {
                    "branch": branch,
                    "step": int(match.group(1)),
                    "test_prompt": str(metric_row["system_prompt_id"]),
                    **{key: float(metric_row[key]) for key in EVAL_METRICS},
                    "prompt_tokens": float(metric_row["prompt_tokens_mean"]),
                    "completion_tokens": float(metric_row["completion_tokens_mean"]),
                    "latency_seconds": float(metric_row["latency_seconds_mean"]),
                }
            )
    if len(rows) != 45:
        raise ValueError(f"Expected 45 checkpoint cells, found {len(rows)}")
    output = []
    for branch in TRAINED_MODELS:
        for step in (20, 40, 60, 80, 100):
            subset = [row for row in rows if row["branch"] == branch and row["step"] == step]
            output.append(
                {
                    "branch": branch,
                    "step": step,
                    **{key: mean(subset, key) for key in (*EVAL_METRICS, *EXTRA_METRICS)},
                }
            )
    return output


def load_training_metrics() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in sorted((ARTIFACTS / "training" / "runs").glob("retool-coding-0812-*/metrics.jsonl")):
        branch = path.parent.name.split("-seed", 1)[0].rsplit("-", 1)[-1]
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if [int(row["step"]) for row in rows] != list(range(1, 101)):
            raise ValueError(f"Non-contiguous training metrics in {path}")
        numeric = sorted(
            {
                key
                for row in rows
                for key, value in row.items()
                if key != "step" and isinstance(value, (int, float))
            }
        )

        def window(start: int, end: int) -> dict[str, float]:
            selected = [row for row in rows if start <= int(row["step"]) <= end]
            return {
                key: mean([row for row in selected if key in row], key)
                for key in numeric
                if any(key in row for row in selected)
            }

        result[branch] = {
            "steps": len(rows),
            "overall": window(1, 100),
            "first20": window(1, 20),
            "last20": window(81, 100),
            "windows20": {
                f"{start}-{start + 19}": window(start, start + 19)
                for start in (1, 21, 41, 61, 81)
            },
        }
    return result


def main() -> None:
    rows = load_final_rows()
    posthoc_rows = load_posthoc_rows()
    all_eval_rows = rows + posthoc_rows
    cell_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cell_groups[f"{row['model'].upper()}×{row['test_prompt']}"] .append(row)
    model_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prompt_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        model_groups[row["model"]].append(row)
        prompt_groups[row["test_prompt"]].append(row)

    output = {
        "provenance": {
            "root": str(ROOT),
            "dataset": str(DATASET),
            "final_rows": len(rows),
            "posthoc_rows": len(posthoc_rows),
            "all_evaluation_rows": len(all_eval_rows),
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "cells": {key: summarize_rows(value) for key, value in sorted(cell_groups.items())},
        "model_marginals": {key: summarize_rows(value) for key, value in sorted(model_groups.items())},
        "prompt_marginals": {key: summarize_rows(value) for key, value in sorted(prompt_groups.items())},
        "primary_vs_base": primary_vs_base(rows),
        "prompt_pair_effects": prompt_pair_effects(rows),
        "model_marginal_contrasts": marginal_contrasts(
            rows,
            condition_key="model",
            conditions=MODELS,
            metrics=(*EVAL_METRICS, *EXTRA_METRICS),
        ),
        "prompt_marginal_contrasts_all_models": marginal_contrasts(
            rows,
            condition_key="test_prompt",
            conditions=PROMPTS,
            metrics=(*EVAL_METRICS, *EXTRA_METRICS),
        ),
        "prompt_marginal_contrasts_trained_only": marginal_contrasts(
            rows,
            condition_key="test_prompt",
            conditions=PROMPTS,
            metrics=(*EVAL_METRICS, *EXTRA_METRICS),
            filters={"model": set(TRAINED_MODELS)},
        ),
        "matching_effect": matching_effect(rows),
        "cap_format_association": cap_format_association(rows),
        "difficulty": stratified_pass(rows, "difficulty"),
        "platform": stratified_pass(rows, "platform"),
        "variance_spans": variance_spans(rows),
        "representative_cases": representative_cases(rows),
        "checkpoint_dev": load_checkpoint_rows(),
        "training": load_training_metrics(),
        "posthoc": {
            "cells": group_summaries(
                posthoc_rows,
                lambda row: f"{row['model'].upper()}×{row['test_prompt']}",
            ),
            "model_marginals": group_summaries(posthoc_rows, lambda row: row["model"]),
            "prompt_pair_effects": prompt_pair_effects(posthoc_rows, POSTHOC_MODELS),
            "model_pair_effects": model_pair_effects(
                all_eval_rows,
                (
                    ("base", "shared-sft-only"),
                    ("shared-sft-only", "c0"),
                    ("shared-sft-only", "c0-step100"),
                    ("shared-sft-only", "c1"),
                    ("shared-sft-only", "c2"),
                    ("c0", "c0-step100"),
                    ("c0-step100", "c1"),
                    ("c0-step100", "c2"),
                ),
            ),
            "model_marginal_contrasts": marginal_contrasts(
                all_eval_rows,
                condition_key="model",
                conditions=ALL_EVAL_MODELS,
                metrics=(*EVAL_METRICS, *EXTRA_METRICS),
            ),
            "all_model_marginals": group_summaries(all_eval_rows, lambda row: row["model"]),
            "all_prompt_marginals": group_summaries(all_eval_rows, lambda row: row["test_prompt"]),
            "cap_format_association": cap_format_association(all_eval_rows),
            "difficulty": stratified_pass(all_eval_rows, "difficulty", ALL_EVAL_MODELS),
            "platform": stratified_pass(all_eval_rows, "platform", ALL_EVAL_MODELS),
        },
    }
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
