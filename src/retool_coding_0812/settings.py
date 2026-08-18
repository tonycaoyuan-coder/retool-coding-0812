"""Load and strictly validate the frozen seed-42 experiment configuration."""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "experiment.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Return the cached config only after every formal value is frozen-valid."""

    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Frozen experiment config must be a mapping")
    _validate_frozen_values(value)
    return value


def resolve_path(value: str) -> Path:
    return (ROOT / value).resolve()


def _validate_frozen_values(config: dict[str, Any]) -> None:
    experiment = config["experiment"]
    sft = config["shared_sft"]
    grpo = config["grpo"]
    protocol = config["protocol"]
    dev = config["checkpoint_dev"]
    evaluation = config["evaluation"]
    swanlab = config["swanlab"]
    expected = {
        "seed": 42,
        "base_model": "Qwen/Qwen3.5-4B",
        "lora_rank": 32,
        "sft_epochs": 3,
        "sft_lr": 1e-5,
        "steps": 100,
        "questions_per_step": 4,
        "group_size": 8,
        "train_temperature": 1.0,
        "train_top_p": 1.0,
        "grpo_lr": 4e-5,
        "save_every": 20,
        "max_tool_calls": 1,
        "max_assistant_turns": 2,
        "max_assistant_tokens": 10240,
        "max_trajectory_tokens": 20480,
        "tool_response_tokens": 512,
        "dev_tasks": 100,
        "checkpoints": [20, 40, 60, 80, 100],
        "test_tasks": 200,
        "eval_temperature": 0.0,
        "swanlab_enabled": True,
        "swanlab_mode": "online",
    }
    actual = {
        "seed": experiment["seed"],
        "base_model": experiment["base_model"],
        "lora_rank": experiment["lora_rank"],
        "sft_epochs": sft["epochs"],
        "sft_lr": float(sft["learning_rate"]),
        "steps": grpo["steps"],
        "questions_per_step": grpo["questions_per_step"],
        "group_size": grpo["group_size"],
        "train_temperature": float(grpo["temperature"]),
        "train_top_p": float(grpo["top_p"]),
        "grpo_lr": float(grpo["learning_rate"]),
        "save_every": grpo["save_every"],
        "max_tool_calls": protocol["max_tool_calls"],
        "max_assistant_turns": protocol["max_assistant_turns"],
        "max_assistant_tokens": protocol["max_assistant_tokens"],
        "max_trajectory_tokens": protocol["max_trajectory_tokens"],
        "tool_response_tokens": protocol["max_tool_response_tokens"],
        "dev_tasks": dev["tasks"],
        "checkpoints": list(dev["checkpoints"]),
        "test_tasks": evaluation["tasks"],
        "eval_temperature": float(evaluation["temperature"]),
        "swanlab_enabled": swanlab["enabled"],
        "swanlab_mode": swanlab["mode"],
    }
    if actual != expected:
        mismatches = {
            key: (actual[key], expected[key])
            for key in expected
            if actual[key] != expected[key]
        }
        raise ValueError(f"Frozen experiment configuration drifted: {mismatches}")
    if list(grpo["variants"]) != ["c0", "c1", "c2"]:
        raise ValueError("The formal experiment requires exactly C0/C1/C2")
    if list(dev["prompts"]) != ["c0", "c1", "c2"]:
        raise ValueError("Checkpoint-dev requires the complete prompt cross")
    if list(evaluation["prompts"]) != ["c0", "c1", "c2"]:
        raise ValueError("Final evaluation requires the complete prompt cross")


def branch_args(
    variant: str,
    *,
    resume_checkpoint: Path | None = None,
    recover: bool = False,
    gate_only: bool = False,
) -> argparse.Namespace:
    """Materialize one branch's complete argument set from the frozen config."""

    config = load_config()
    experiment = config["experiment"]
    inputs = config["inputs"]
    grpo = config["grpo"]
    protocol = config["protocol"]
    docker = config["docker"]
    swanlab = config["swanlab"]
    if variant not in grpo["variants"]:
        raise ValueError(f"Unknown branch {variant!r}")
    return argparse.Namespace(
        prompt_variant=variant,
        data=resolve_path(f"{inputs['formal_data_dir']}/train.jsonl.gz"),
        base_model=experiment["base_model"],
        resume_checkpoint=resume_checkpoint,
        recover=recover,
        initial_state_manifest=(
            None
            if resume_checkpoint is not None
            else (
                ROOT / "inputs/shared-sft-e3-seed42.json"
                if gate_only
                else ROOT / "artifacts/training/shared-sft/seed42/manifest.json"
            )
        ),
        lora_rank=experiment["lora_rank"],
        seed=experiment["seed"],
        max_steps=grpo["steps"],
        questions_per_batch=grpo["questions_per_step"],
        group_size=grpo["group_size"],
        max_tool_calls=protocol["max_tool_calls"],
        max_assistant_turns=protocol["max_assistant_turns"],
        max_trajectory_tokens=protocol["max_trajectory_tokens"],
        max_assistant_tokens=protocol["max_assistant_tokens"],
        max_tool_response_tokens=protocol["max_tool_response_tokens"],
        temperature=grpo["temperature"],
        top_p=grpo["top_p"],
        learning_rate=grpo["learning_rate"],
        beta1=grpo["beta1"],
        beta2=grpo["beta2"],
        max_micro_batch_items=grpo["max_micro_batch_items"],
        max_micro_batch_padded_tokens=grpo["max_micro_batch_padded_tokens"],
        save_every=grpo["save_every"],
        max_step_retries=grpo["max_step_retries"],
        early_gate_step=grpo["early_gate_step"],
        min_early_nondegenerate_rate=grpo["min_nondegenerate_group_rate"],
        max_early_skipped_update_rate=grpo["max_skipped_update_rate"],
        docker_binary=docker["binary"],
        image=docker["image"],
        preflight_manifest=resolve_path(inputs["docker_preflight"]),
        data_manifest=resolve_path(inputs["data_manifest"]),
        smoke_manifest=resolve_path(inputs["calibration_gate"]),
        protocol_config=resolve_path(inputs["selected_protocol"]),
        sandbox_workers=docker["workers"],
        run_name=f"retool-coding-0812-{variant}-seed42",
        gate_only=gate_only,
        checkpoint_dir=ROOT / "artifacts/training/checkpoints",
        artifact_dir=ROOT / "artifacts/training/runs",
        swanlab_project=swanlab["project"],
        swanlab_group=swanlab["group"],
        swanlab_mode=swanlab["mode"],
    )
