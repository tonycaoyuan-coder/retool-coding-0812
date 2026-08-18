"""Local compressed trajectory artifacts without hidden test material."""

from __future__ import annotations

import gzip
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .rollout import Trajectory


def trajectory_record(trajectory: Trajectory, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    judge_result = asdict(trajectory.judge_result) if trajectory.judge_result else None
    if judge_result is not None:
        # ``resolved`` is a derived JudgeResult property, so dataclasses.asdict()
        # does not persist it.  Store it explicitly because downstream artifact
        # validation must distinguish a completed judge call from an all-tests
        # pass without reconstructing an SDK/runtime object.
        judge_result["resolved"] = trajectory.judge_result.resolved
    return {
        "metadata": metadata or {},
        "example": trajectory.example.prompt_record(),
        "question_index": trajectory.question_index,
        "group_index": trajectory.group_index,
        "messages": trajectory.messages,
        "turns": [asdict(turn) for turn in trajectory.turns],
        "tool_calls": trajectory.tool_calls,
        "tool_call_attempts": trajectory.tool_call_attempts,
        "valid_tool_calls": trajectory.valid_tool_calls,
        "final_text": trajectory.final_text,
        "final_code": trajectory.final_code,
        "judge_result": judge_result,
        "reward": trajectory.reward,
        "advantage": trajectory.advantage,
        "duration_seconds": trajectory.duration_seconds,
        "trajectory_budget_exhausted": trajectory.trajectory_budget_exhausted,
        "hit_token_limit": trajectory.hit_token_limit,
        "finish_reason": trajectory.finish_reason,
    }


def save_trajectory_gzip(
    path: str | Path, trajectory: Trajectory, metadata: dict[str, Any] | None = None
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(destination, "wt", encoding="utf-8") as stream:
        json.dump(trajectory_record(trajectory, metadata), stream, ensure_ascii=False)


def aggregate_behavior_metrics(trajectories: list[Trajectory]) -> dict[str, float]:
    if not trajectories:
        return {}
    count = len(trajectories)
    attempted = sum(item.tool_call_attempts for item in trajectories)
    return {
        "tool_use_rate": sum(item.tool_calls > 0 for item in trajectories) / count,
        "mean_tool_calls": sum(item.tool_calls for item in trajectories) / count,
        "tool_call_valid_rate": sum(item.valid_tool_calls for item in trajectories) / max(attempted, 1),
        "format_valid_rate": sum(item.final_code is not None for item in trajectories) / count,
        "token_cap_hit_rate": sum(item.hit_token_limit for item in trajectories) / count,
        "mean_turns": sum(len(item.turns) for item in trajectories) / count,
        "mean_trajectory_tokens": sum(
            len(item.turns[-1].prompt_tokens) + len(item.turns[-1].completion_tokens)
            for item in trajectories if item.turns
        ) / count,
        "mean_execution_seconds": sum(
            item.judge_result.execution_seconds if item.judge_result else 0.0
            for item in trajectories
        ) / count,
        "mean_agent_seconds": sum(item.duration_seconds for item in trajectories) / count,
    }
