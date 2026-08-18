"""Audit completeness and provenance of the L16K/T24K final evaluation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import sqlite3
from typing import Any

import yaml


MODELS = ("base", "c0", "c1", "c2")
PROMPTS = ("c0", "c1", "c2")
REQUIRED_RUN_FILES = (
    "manifest.json",
    "resolved_config.yaml",
    "state.sqlite",
    "summary/metrics.json",
    "summary/metrics.csv",
    "report/index.html",
)
REQUIRED_TRAJECTORY_KEYS = {
    "metadata",
    "example",
    "question_index",
    "group_index",
    "messages",
    "turns",
    "tool_calls",
    "tool_call_attempts",
    "valid_tool_calls",
    "final_text",
    "final_code",
    "judge_result",
    "reward",
    "advantage",
    "duration_seconds",
    "trajectory_budget_exhausted",
    "hit_token_limit",
    "finish_reason",
}
REQUIRED_TURN_KEYS = {
    "prompt_tokens",
    "completion_tokens",
    "logprobs",
    "text",
    "stop_reason",
    "requested_max_tokens",
    "completion_token_count",
    "hit_token_limit",
}
REQUIRED_JUDGE_KEYS = {
    "passed",
    "total",
    "public_passed",
    "public_total",
    "private_passed",
    "private_total",
    "status",
    "first_failure",
    "execution_seconds",
    "infrastructure_error",
    "details",
    "resolved",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--myeval-runs", nargs=3, type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--baseline-config-dir", type=Path, required=True)
    parser.add_argument("--tasks", type=int, default=200)
    parser.add_argument("--max-assistant-tokens", type=int, default=16384)
    parser.add_argument("--max-trajectory-tokens", type=int, default=24576)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def prompt_id(config: dict[str, Any]) -> str:
    profiles = list(config["system_prompts"]["profiles"])
    if len(profiles) != 1:
        raise ValueError("Each final-evaluation shard must contain one system prompt")
    prompt = str(profiles[0]["id"]).lower()
    if prompt not in PROMPTS:
        raise ValueError(f"Unexpected prompt shard {prompt!r}")
    return prompt


def comparable_config(config: dict[str, Any]) -> dict[str, Any]:
    benchmark = dict(config["benchmarks"][0])
    options = dict(benchmark["options"])
    options.pop("max_trajectory_tokens", None)
    benchmark["options"] = options
    generation = dict(config["generation"])
    generation.pop("max_out_length", None)
    execution = dict(config["execution"])
    execution.pop("output_dir", None)
    execution.pop("timeout_seconds", None)
    tracking = dict(config["tracking"])
    tracking.pop("experiment_name", None)
    tracking.pop("job_type", None)
    models = []
    for model in config["models"]:
        item = dict(model)
        metadata = dict(item.get("metadata") or {})
        metadata.pop("evaluation_role", None)
        item["metadata"] = metadata
        models.append(item)
    return {
        "seed": config["experiment"]["seed"],
        "models": models,
        "benchmark": benchmark,
        "system_prompts": config["system_prompts"],
        "generation": generation,
        "execution": execution,
        "evaluation": config["evaluation"],
        "tracking": tracking,
    }


def audit_config_pair(
    new_path: Path,
    baseline_path: Path,
    max_assistant_tokens: int,
    max_trajectory_tokens: int,
    timeout_seconds: int,
) -> str:
    new = load_yaml(new_path)
    baseline = load_yaml(baseline_path)
    prompt = prompt_id(new)
    if prompt_id(baseline) != prompt:
        raise ValueError(f"Prompt mismatch between {new_path} and {baseline_path}")
    if comparable_config(new) != comparable_config(baseline):
        raise ValueError(f"Non-length evaluation configuration drifted for {prompt}")
    if int(new["generation"]["max_out_length"]) != max_assistant_tokens:
        raise ValueError(f"Wrong assistant token cap in {new_path}")
    observed_trajectory = int(new["benchmarks"][0]["options"]["max_trajectory_tokens"])
    if observed_trajectory != max_trajectory_tokens:
        raise ValueError(f"Wrong trajectory token cap in {new_path}")
    if int(new["execution"]["timeout_seconds"]) != timeout_seconds:
        raise ValueError(f"Wrong execution timeout in {new_path}")
    return prompt


def validate_trajectory(path: Path, expected_prompt: str) -> tuple[str, str]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        row = json.load(stream)
    missing = REQUIRED_TRAJECTORY_KEYS - set(row)
    if missing:
        raise ValueError(f"{path}: missing trajectory keys {sorted(missing)}")
    metadata = dict(row["metadata"])
    model = str(metadata.get("model_id", "")).lower()
    if model not in MODELS:
        raise ValueError(f"{path}: unexpected model metadata {model!r}")
    if str(metadata.get("system_prompt_id", "")).lower() != expected_prompt:
        raise ValueError(f"{path}: prompt metadata mismatch")
    if not metadata.get("task_key"):
        raise ValueError(f"{path}: missing task_key")
    example = dict(row["example"])
    forbidden = {"all_tests", "public_tests", "private_tests"} & set(example)
    if forbidden:
        raise ValueError(f"{path}: hidden test material leaked into trajectory: {sorted(forbidden)}")
    sample_id = str(example.get("instance_id") or "")
    if not sample_id:
        raise ValueError(f"{path}: missing instance_id")
    messages = list(row["messages"])
    roles = [str(item.get("role", "")) for item in messages]
    if roles[:2] != ["system", "user"] or "assistant" not in roles:
        raise ValueError(f"{path}: incomplete message history {roles}")
    if int(row["tool_calls"]) > 0 and "tool" not in roles:
        raise ValueError(f"{path}: executed tool call has no persisted observation")
    turns = list(row["turns"])
    if not turns:
        raise ValueError(f"{path}: no assistant turns")
    for index, turn in enumerate(turns):
        missing_turn = REQUIRED_TURN_KEYS - set(turn)
        if missing_turn:
            raise ValueError(f"{path}: turn {index} missing keys {sorted(missing_turn)}")
        prompt_tokens = list(turn["prompt_tokens"])
        completion_tokens = list(turn["completion_tokens"])
        logprobs = list(turn["logprobs"])
        count = int(turn["completion_token_count"])
        if len(completion_tokens) != len(logprobs) or len(completion_tokens) != count:
            raise ValueError(f"{path}: turn {index} token/logprob/count mismatch")
        if not prompt_tokens:
            raise ValueError(f"{path}: turn {index} has no prompt tokens")
        if int(turn["requested_max_tokens"]) < count:
            raise ValueError(f"{path}: turn {index} exceeds requested token count")
        if not isinstance(turn["text"], str) or not isinstance(turn["stop_reason"], str):
            raise ValueError(f"{path}: turn {index} text/stop reason type mismatch")
    judge = dict(row["judge_result"] or {})
    missing_judge = REQUIRED_JUDGE_KEYS - set(judge)
    if missing_judge:
        raise ValueError(f"{path}: missing judge keys {sorted(missing_judge)}")
    return model, sample_id


def audit_run(run_dir: Path, tasks: int) -> dict[str, Any]:
    for relative in REQUIRED_RUN_FILES:
        if not (run_dir / relative).is_file():
            raise ValueError(f"{run_dir}: missing MyEval artifact {relative}")
    if not (run_dir / "logs").is_dir() or not (run_dir / "swanlog").is_dir():
        raise ValueError(f"{run_dir}: missing logs or SwanLab artifact directory")
    resolved = load_yaml(run_dir / "resolved_config.yaml")
    prompt = prompt_id(resolved)
    counts = json.loads((run_dir / "summary/metrics.json").read_text(encoding="utf-8"))["counts"]
    expected_total = len(MODELS) * tasks
    if int(counts.get("completed", 0)) != expected_total or any(
        int(counts.get(status, 0)) for status in ("failed", "pending", "running")
    ):
        raise ValueError(f"{run_dir}: incomplete MyEval counts {counts}")
    with sqlite3.connect(run_dir / "state.sqlite") as connection:
        state_rows = connection.execute(
            "SELECT model_id, sample_id, status FROM tasks"
        ).fetchall()
    if len(state_rows) != expected_total or any(status != "completed" for _, _, status in state_rows):
        raise ValueError(f"{run_dir}: state.sqlite is incomplete")
    expected_pairs = {(str(model), str(sample)) for model, sample, _ in state_rows}
    by_model: dict[str, set[str]] = defaultdict(set)
    for model, sample in expected_pairs:
        by_model[model].add(sample)
    if set(by_model) != set(MODELS) or any(len(values) != tasks for values in by_model.values()):
        raise ValueError(f"{run_dir}: model/sample pairing is incomplete")
    if len({frozenset(values) for values in by_model.values()}) != 1:
        raise ValueError(f"{run_dir}: models do not share identical sample IDs")

    artifact_paths = sorted((run_dir / "artifacts/trajectories").glob("*.json.gz"))
    if len(artifact_paths) != expected_total:
        raise ValueError(f"{run_dir}: expected {expected_total} trajectories, found {len(artifact_paths)}")
    observed_pairs: set[tuple[str, str]] = set()
    model_counts: Counter[str] = Counter()
    for path in artifact_paths:
        model, sample_id = validate_trajectory(path, prompt)
        pair = (model, sample_id)
        if pair in observed_pairs:
            raise ValueError(f"{run_dir}: duplicate trajectory pair {pair}")
        observed_pairs.add(pair)
        model_counts[model] += 1
    if observed_pairs != expected_pairs:
        raise ValueError(f"{run_dir}: trajectory and SQLite task sets differ")
    return {
        "prompt": prompt,
        "run_dir": str(run_dir.resolve()),
        "completed": expected_total,
        "models": dict(sorted(model_counts.items())),
        "sample_ids": sorted(next(iter(by_model.values()))),
        "trajectory_schema": "complete",
    }


def main(args: argparse.Namespace) -> None:
    config_prompts = {
        audit_config_pair(
            args.config_dir / f"{prompt}.yaml",
            args.baseline_config_dir / f"{prompt}.yaml",
            args.max_assistant_tokens,
            args.max_trajectory_tokens,
            args.timeout_seconds,
        )
        for prompt in PROMPTS
    }
    if config_prompts != set(PROMPTS):
        raise ValueError(f"Config prompt set is incomplete: {sorted(config_prompts)}")
    runs = [audit_run(path, args.tasks) for path in args.myeval_runs]
    if {row["prompt"] for row in runs} != set(PROMPTS):
        raise ValueError("Run prompt set is incomplete or duplicated")
    sample_sets = {frozenset(row["sample_ids"]) for row in runs}
    if len(sample_sets) != 1:
        raise ValueError("Prompt shards do not share identical sample IDs")
    result = {
        "status": "complete",
        "max_assistant_tokens": args.max_assistant_tokens,
        "max_trajectory_tokens": args.max_trajectory_tokens,
        "execution_timeout_seconds": args.timeout_seconds,
        "tasks_per_cell": args.tasks,
        "cells": len(MODELS) * len(PROMPTS),
        "completed": len(MODELS) * len(PROMPTS) * args.tasks,
        "runs": sorted(runs, key=lambda item: item["prompt"]),
        "hidden_tests_persisted_in_trajectories": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main(parse_args())
