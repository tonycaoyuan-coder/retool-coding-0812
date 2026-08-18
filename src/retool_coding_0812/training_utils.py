"""PyTRIO importance-sampling datum construction and training metrics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytrio as trio

from .rollout import Trajectory


@dataclass(frozen=True)
class TrainingDatum:
    datum: trio.Datum
    num_tokens: int


class TrainingContextLengthError(ValueError):
    """A rollout is valid for evaluation but too long for the trainer backend."""


def build_datum(trajectory: Trajectory, *, max_context_tokens: int = 8192) -> TrainingDatum:
    """Convert one rollout into a right-shifted PyTRIO GRPO datum.

    Prompt and tool-observation tokens are context only, so their old logprobs
    and advantages are zero. Assistant completion tokens retain rollout-time
    logprobs and receive the trajectory's group-relative advantage.
    """

    if not trajectory.turns:
        raise ValueError("Cannot train on an empty trajectory")
    full_tokens: list[int] = []
    old_logprobs: list[float] = []
    advantages: list[float] = []
    for index, turn in enumerate(trajectory.turns):
        if len(turn.completion_tokens) != len(turn.logprobs):
            raise ValueError("Completion tokens and logprobs are not aligned")
        if index == 0:
            observation = turn.prompt_tokens
        elif turn.prompt_tokens[: len(full_tokens)] == full_tokens:
            observation = turn.prompt_tokens[len(full_tokens) :]
        else:
            raise ValueError("Later prompt is not a prefix extension of the trajectory")
        full_tokens.extend(observation)
        full_tokens.extend(turn.completion_tokens)
        old_logprobs.extend([0.0] * len(observation))
        old_logprobs.extend(turn.logprobs)
        advantages.extend([0.0] * len(observation))
        advantages.extend([trajectory.advantage] * len(turn.completion_tokens))
    inputs = full_tokens[:-1]
    targets = full_tokens[1:]
    shifted_logprobs = old_logprobs[1:]
    shifted_advantages = advantages[1:]
    if len({len(inputs), len(targets), len(shifted_logprobs), len(shifted_advantages)}) != 1:
        raise ValueError("Autoregressive fields are misaligned")
    if len(inputs) > max_context_tokens:
        raise TrainingContextLengthError("Trajectory exceeds training context budget")
    return TrainingDatum(
        datum=trio.Datum(
            model_input=trio.ModelInput.from_ints(inputs),
            loss_fn_inputs={
                "target_tokens": np.asarray(targets, dtype=np.int64),
                "logprobs": np.asarray(shifted_logprobs, dtype=np.float32),
                "advantages": np.asarray(shifted_advantages, dtype=np.float32),
            },
        ),
        num_tokens=len(inputs),
    )


def build_training_datums(
    trajectories: list[Trajectory], *, max_context_tokens: int = 8192
) -> list[TrainingDatum]:
    """Build trainable datums while retaining skipped rollouts in telemetry."""

    datums: list[TrainingDatum] = []
    for item in trajectories:
        if item.advantage == 0.0 or not item.turns:
            continue
        try:
            datums.append(build_datum(item, max_context_tokens=max_context_tokens))
        except TrainingContextLengthError:
            # The rollout remains part of reward/gate telemetry, but the remote
            # trainer cannot accept a sequence beyond its advertised hard cap.
            continue
    return datums


def pack_micro_batches(
    datums: list[TrainingDatum], *, max_items: int = 16, max_padded_tokens: int = 65_536
) -> list[list[TrainingDatum]]:
    """Greedily pack datums under item-count and padded-token limits."""

    batches: list[list[TrainingDatum]] = []
    maxima: list[int] = []
    for item in sorted(datums, key=lambda value: value.num_tokens, reverse=True):
        for index, batch in enumerate(batches):
            count = len(batch) + 1
            maximum = max(maxima[index], item.num_tokens)
            if count <= max_items and count * maximum <= max_padded_tokens:
                batch.append(item)
                maxima[index] = maximum
                break
        else:
            if item.num_tokens > max_padded_tokens:
                raise ValueError("One datum exceeds the padded-token budget")
            batches.append([item])
            maxima.append(item.num_tokens)
    return batches


def weight_micro_batch_for_global_mean(
    micro_batch: list[TrainingDatum], total_trajectories: int
) -> list[trio.Datum]:
    """Scale each micro-batch so accumulation equals the full-step mean loss."""

    weight = np.float32(len(micro_batch) / total_trajectories)
    output = []
    for item in micro_batch:
        fields = item.datum.loss_fn_inputs
        output.append(
            trio.Datum(
                model_input=item.datum.model_input,
                loss_fn_inputs={
                    "target_tokens": fields["target_tokens"].to_numpy(),
                    "logprobs": fields["logprobs"].to_numpy(),
                    "advantages": fields["advantages"].to_numpy() * weight,
                },
            )
        )
    return output


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def rollout_metrics(
    trajectories: list[Trajectory],
    datums: list[TrainingDatum],
    micro_batches: list[list[TrainingDatum]],
    question_count: int,
) -> dict[str, float]:
    judges = [item.judge_result for item in trajectories if item.judge_result is not None]
    attempted = sum(item.tool_call_attempts for item in trajectories)
    grouped: dict[int, list[float]] = defaultdict(list)
    for item in trajectories:
        grouped[item.question_index].append(item.reward)
    status_names = ("compile_error", "runtime_error", "time_limit", "invalid_format")
    metrics = {
        "reward/mean": _mean([item.reward for item in trajectories]),
        "reward/pass_at_1": _mean([float(item.resolved) for item in judges]),
        "reward/case_pass_rate": _mean([item.passed / max(item.total, 1) for item in judges]),
        "reward/public_pass_rate": _mean([item.public_rate for item in judges]),
        "reward/private_pass_rate": _mean([item.private_rate for item in judges]),
        "rollout/format_valid_rate": _mean([float(item.final_code is not None) for item in trajectories]),
        "rollout/token_cap_hit_rate": _mean([float(item.hit_token_limit) for item in trajectories]),
        "rollout/tool_use_rate": _mean([float(item.tool_calls > 0) for item in trajectories]),
        "rollout/mean_tool_calls": _mean([float(item.tool_calls) for item in trajectories]),
        "rollout/valid_tool_call_rate": sum(item.valid_tool_calls for item in trajectories) / max(attempted, 1),
        "rollout/mean_turns": _mean([float(len(item.turns)) for item in trajectories]),
        "rollout/mean_tokens": _mean([
            float(len(item.turns[-1].prompt_tokens) + len(item.turns[-1].completion_tokens))
            for item in trajectories if item.turns
        ]),
        "rollout/degenerate_group_rate": sum(len(set(values)) == 1 for values in grouped.values()) / max(question_count, 1),
        "train/datums_per_step": float(len(datums)),
        "train/micro_batches_per_step": float(len(micro_batches)),
        "train/update_skipped": float(not micro_batches),
    }
    for status in status_names:
        metrics[f"judge/{status}_rate"] = _mean([float(item.status == status) for item in judges])
    return metrics


def merge_trainer_metrics(results: list[Any]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for result in results:
        for key, value in dict(result.metrics).items():
            if isinstance(value, (int, float, np.number)):
                grouped[key].append(float(value))
    return {f"trainer/{key}": _mean(values) for key, values in grouped.items()}


def early_training_signal_gate(
    rows: list[dict[str, float]],
    *,
    min_nondegenerate_group_rate: float = 0.50,
    max_skipped_update_rate: float = 0.25,
) -> dict[str, float | bool | int]:
    """Check that early rollout groups provide enough non-zero update signal."""

    if not rows:
        raise ValueError("Early training gate requires metric rows")
    nondegenerate = 1.0 - _mean(
        [float(row["rollout/degenerate_group_rate"]) for row in rows]
    )
    skipped = _mean([float(row["train/update_skipped"]) for row in rows])
    return {
        "passed": (
            nondegenerate >= min_nondegenerate_group_rate
            and skipped <= max_skipped_update_rate
        ),
        "steps": len(rows),
        "nondegenerate_group_rate": nondegenerate,
        "skipped_update_rate": skipped,
        "min_nondegenerate_group_rate": min_nondegenerate_group_rate,
        "max_skipped_update_rate": max_skipped_update_rate,
    }
