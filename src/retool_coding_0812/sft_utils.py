"""Build assistant-only SFT datums from the frozen validated trajectories."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytrio as trio

from .data import file_sha256
from .training_utils import TrainingDatum


def _judge_resolved(judge: dict[str, Any]) -> bool:
    if "resolved" in judge:
        return bool(judge["resolved"])
    try:
        total = int(judge.get("total", 0))
        passed = int(judge.get("passed", 0))
    except (TypeError, ValueError):
        return False
    return total > 0 and passed == total and not judge.get("infrastructure_error")


def build_sft_datum(
    record: dict[str, Any], *, max_context_tokens: int = 12288
) -> TrainingDatum:
    """Build a right-shifted, assistant-only SFT datum from a proven trajectory.

    Only trajectories that passed every execution test, produced strict final
    code, and avoided token caps are accepted. Prompt and tool observations use
    zero loss weight; assistant completion tokens use weight one.
    """

    turns = list(record.get("turns") or [])
    judge = dict(record.get("judge_result") or {})
    if not turns:
        raise ValueError("SFT trajectory has no assistant turns")
    if not _judge_resolved(judge):
        raise ValueError("SFT trajectory did not pass every execution test")
    if bool(record.get("hit_token_limit", False)):
        raise ValueError("SFT trajectory hit a token cap")
    if not record.get("final_code"):
        raise ValueError("SFT trajectory is not a strict final code response")
    full_tokens: list[int] = []
    weights: list[float] = []
    for index, raw in enumerate(turns):
        prompt_tokens = [int(value) for value in raw["prompt_tokens"]]
        completion_tokens = [int(value) for value in raw["completion_tokens"]]
        if int(raw.get("completion_token_count", len(completion_tokens))) != len(
            completion_tokens
        ):
            raise ValueError("SFT completion token count is inconsistent")
        if bool(raw.get("hit_token_limit", False)):
            raise ValueError("SFT assistant turn hit a token cap")
        if index == 0:
            observation = prompt_tokens
        elif prompt_tokens[: len(full_tokens)] == full_tokens:
            observation = prompt_tokens[len(full_tokens) :]
        else:
            raise ValueError("Later SFT prompt is not a prefix extension")
        full_tokens.extend(observation)
        full_tokens.extend(completion_tokens)
        weights.extend([0.0] * len(observation))
        weights.extend([1.0] * len(completion_tokens))
    inputs = full_tokens[:-1]
    targets = full_tokens[1:]
    shifted_weights = weights[1:]
    if len({len(inputs), len(targets), len(shifted_weights)}) != 1:
        raise ValueError("SFT autoregressive fields are misaligned")
    if not inputs or len(inputs) > max_context_tokens:
        raise ValueError("SFT trajectory exceeds context budget or is empty")
    if sum(shifted_weights) <= 0:
        raise ValueError("SFT trajectory has no assistant loss tokens")
    return TrainingDatum(
        datum=trio.Datum(
            model_input=trio.ModelInput.from_ints(inputs),
            loss_fn_inputs={
                "target_tokens": np.asarray(targets, dtype=np.int32),
                "weights": np.asarray(shifted_weights, dtype=np.float32),
            },
        ),
        num_tokens=len(inputs),
    )


def load_validated_sft_datums(
    manifest_path: str | Path, *, max_context_tokens: int = 12288
) -> tuple[list[TrainingDatum], dict[str, Any]]:
    """Load exactly 300 unique, hash-verified SFT source trajectories."""

    source = Path(manifest_path)
    manifest = json.loads(source.read_text(encoding="utf-8"))
    selected = list(manifest.get("selected") or [])
    if not manifest.get("complete") or len(selected) != 300:
        raise ValueError("Shared SFT manifest must contain exactly 300 complete records")
    datums = []
    identifiers = set()
    for item in selected:
        instance_id = str(item["instance_id"])
        if instance_id in identifiers:
            raise ValueError(f"Duplicate shared SFT task {instance_id}")
        identifiers.add(instance_id)
        artifact = source.parent / str(item["artifact"])
        if file_sha256(artifact) != item.get("artifact_sha256"):
            raise ValueError(f"Shared SFT artifact SHA-256 mismatch: {artifact}")
        with gzip.open(artifact, "rt", encoding="utf-8") as stream:
            record = json.load(stream)
        if str((record.get("example") or {}).get("instance_id")) != instance_id:
            raise ValueError(f"Shared SFT artifact id mismatch: {artifact}")
        datums.append(build_sft_datum(record, max_context_tokens=max_context_tokens))
    return datums, manifest
