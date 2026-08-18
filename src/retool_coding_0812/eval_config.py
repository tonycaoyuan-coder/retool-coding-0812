"""Build MyEval configs from frozen checkpoints and experiment settings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

import yaml

from .protocol import system_prompt
from .settings import ROOT, load_config, resolve_path


def _models(manifest_path: Path) -> dict[str, dict[str, Any] | None]:
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Model manifest must be a JSON object")
    return value


def _model_entry(label: str, item: dict[str, Any] | None, role: str) -> dict[str, Any]:
    config = load_config()
    experiment = config["experiment"]
    variant = "base" if item is None else str(item["train_variant"]).lower()
    step = 0 if item is None else int(item["step"])
    if item is not None and int(item["seed"]) != 42:
        raise ValueError(f"Only seed 42 is allowed: {label}")
    return {
        "id": label,
        "backend": "pytrio",
        "base_model": experiment["base_model"],
        "model_path": None if item is None else item["model_path"],
        "metadata": {
            "train_system_prompt": variant,
            "checkpoint_step": step,
            "evaluation_role": role,
            "seed": None if item is None else 42,
        },
    }


def _base_config(
    *,
    name: str,
    data: Path,
    tasks: int,
    models: list[dict[str, Any]],
    role: str,
    output_dir: Path,
    prompt_variants: tuple[str, ...] = ("c0", "c1", "c2"),
    max_assistant_tokens: int | None = None,
    max_trajectory_tokens: int | None = None,
    tracking_job_type: str | None = None,
    execution_timeout_seconds: int | None = None,
) -> dict[str, Any]:
    config = load_config()
    experiment = config["experiment"]
    protocol = config["protocol"]
    docker = config["docker"]
    evaluation = config["evaluation"]
    swanlab = config["swanlab"]
    assistant_tokens = (
        protocol["max_assistant_tokens"]
        if max_assistant_tokens is None
        else int(max_assistant_tokens)
    )
    trajectory_tokens = (
        protocol["max_trajectory_tokens"]
        if max_trajectory_tokens is None
        else int(max_trajectory_tokens)
    )
    if assistant_tokens < 1:
        raise ValueError("max_assistant_tokens must be positive")
    if trajectory_tokens < assistant_tokens:
        raise ValueError("max_trajectory_tokens must be at least max_assistant_tokens")
    job_type = tracking_job_type or role
    if len(job_type) > 20:
        raise ValueError("SwanLab tracking job_type must contain at most 20 characters")
    task_timeout = (
        evaluation["timeout_seconds"]
        if execution_timeout_seconds is None
        else int(execution_timeout_seconds)
    )
    if task_timeout < 1:
        raise ValueError("execution_timeout_seconds must be positive")
    return {
        "version": 1,
        "experiment": {
            "name": name,
            "seed": experiment["seed"],
            "description": "Frozen seed-42 ReTool coding formal experiment.",
        },
        "models": models,
        "benchmarks": [
            {
                "id": "lcb_codegen_retool_0812",
                "local_path": str(data.resolve()),
                "shots": [0],
                "limit": tasks,
                "options": {
                    "docker_binary": docker["binary"],
                    "image": docker["image"],
                    "max_tool_calls": protocol["max_tool_calls"],
                    "max_assistant_turns": protocol["max_assistant_turns"],
                    "max_trajectory_tokens": trajectory_tokens,
                    "max_tool_response_tokens": protocol["max_tool_response_tokens"],
                    "tool_timeout_seconds": docker["tool_timeout_seconds"],
                    "case_timeout_seconds": docker["case_timeout_seconds"],
                    "judge_timeout_seconds": docker["judge_timeout_seconds"],
                },
            }
        ],
        "system_prompts": {
            "mode": "global",
            "profiles": [
                {
                    "id": variant.upper(),
                    "text": system_prompt(
                        variant, max_tool_calls=protocol["max_tool_calls"]
                    ),
                }
                for variant in prompt_variants
            ],
        },
        "generation": {
            "temperature": evaluation["temperature"],
            "max_out_length": assistant_tokens,
            "top_p": evaluation["top_p"],
            "top_k": evaluation["top_k"],
            "seed": experiment["seed"],
            "stop": [],
            "n": evaluation["n"],
            "pass_k": evaluation["pass_k"],
        },
        "execution": {
            "output_dir": str(output_dir.resolve()),
            "max_concurrency": evaluation["max_concurrency"],
            "timeout_seconds": task_timeout,
            "retries": evaluation["retries"],
            "retry_base_seconds": evaluation["retry_base_seconds"],
            "infrastructure_failure_threshold": 4,
        },
        "evaluation": {
            "bootstrap_samples": evaluation["bootstrap_samples"],
            "confidence_level": evaluation["confidence_level"],
        },
        "tracking": {
            "swanlab": swanlab["enabled"],
            "project": swanlab["project"],
            "experiment_name": name,
            "mode": swanlab["mode"],
            "group": swanlab["group"],
            "job_type": job_type,
        },
    }


def checkpoint_dev_config(variant: str, manifest_path: Path) -> dict[str, Any]:
    config = load_config()
    required = set(config["checkpoint_dev"]["checkpoints"])
    manifest = _models(manifest_path)
    selected = []
    for label, item in sorted(manifest.items()):
        if item is None or str(item.get("train_variant")).lower() != variant:
            continue
        if int(item.get("step", 0)) in required:
            selected.append(_model_entry(label, item, "checkpoint-dev"))
    steps = {int(item["metadata"]["checkpoint_step"]) for item in selected}
    if steps != required or len(selected) != len(required):
        raise ValueError(f"Incomplete {variant} checkpoint set: {sorted(steps)}")
    return _base_config(
        name=f"retool-coding-0812-checkpoint-dev-{variant}-seed42",
        data=resolve_path(f"{config['inputs']['formal_data_dir']}/checkpoint_dev.jsonl.gz"),
        tasks=config["checkpoint_dev"]["tasks"],
        models=selected,
        role=f"checkpoint-dev-{variant}",
        output_dir=ROOT / "artifacts/checkpoint_dev" / variant,
    )


def final_config(
    manifest_path: Path,
    prompt_variant: str | None = None,
    *,
    max_assistant_tokens: int | None = None,
    max_trajectory_tokens: int | None = None,
    experiment_tag: str | None = None,
    output_root: Path | None = None,
    tasks: int | None = None,
    execution_timeout_seconds: int | None = None,
) -> dict[str, Any]:
    config = load_config()
    manifest = _models(manifest_path)
    expected = {"base", "c0", "c1", "c2"}
    if set(manifest) != expected:
        raise ValueError(f"Final manifest requires {sorted(expected)}, got {sorted(manifest)}")
    if prompt_variant is not None and prompt_variant not in {"c0", "c1", "c2"}:
        raise ValueError(f"Unknown final-evaluation prompt variant: {prompt_variant}")
    tag = str(experiment_tag or "").strip().lower()
    if tag and not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", tag):
        raise ValueError(f"Invalid experiment tag: {experiment_tag!r}")
    task_count = config["evaluation"]["tasks"] if tasks is None else int(tasks)
    if task_count < 1 or task_count > config["evaluation"]["tasks"]:
        raise ValueError(
            f"Final evaluation tasks must be in [1, {config['evaluation']['tasks']}]"
        )
    role_base = f"evaluation-{tag}" if tag else "evaluation"
    role = f"{role_base}-{prompt_variant}" if prompt_variant else role_base
    suffix = f"-{prompt_variant}" if prompt_variant is not None else ""
    tracking_role = f"eval-{tag}{suffix}" if tag else role
    if len(tracking_role) > 20:
        digest = hashlib.sha256(tag.encode("utf-8")).hexdigest()[:4]
        compact_tag = tag[:8].rstrip("-")
        tracking_role = f"ev-{compact_tag}-{digest}{suffix}"
    models = [
        _model_entry(label, manifest[label], role_base)
        for label in ("base", "c0", "c1", "c2")
    ]
    name_tag = f"-{tag}" if tag else ""
    destination_root = output_root or (
        ROOT / f"artifacts/evaluation-{tag}" if tag else ROOT / "artifacts/evaluation"
    )
    return _base_config(
        name=f"retool-coding-0812-evaluation{name_tag}{suffix}-seed42",
        data=resolve_path(f"{config['inputs']['formal_data_dir']}/test.jsonl.gz"),
        tasks=task_count,
        models=models,
        role=role,
        output_dir=destination_root / prompt_variant if prompt_variant else destination_root,
        prompt_variants=(prompt_variant,) if prompt_variant else ("c0", "c1", "c2"),
        max_assistant_tokens=max_assistant_tokens,
        max_trajectory_tokens=max_trajectory_tokens,
        tracking_job_type=tracking_role,
        execution_timeout_seconds=execution_timeout_seconds,
    )


def write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
