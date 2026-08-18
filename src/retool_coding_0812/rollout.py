"""Multi-turn ReTool rollout over stateless run_python calls."""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable

import pytrio as trio

from .data import LCBExample
from .docker_executor import DockerExecutor, DockerInfrastructureError, ToolResult
from .protocol import (
    FINAL_CODE_PREFIX,
    FINAL_SUBMISSION_REMINDER,
    build_next_prompt,
    build_prompt,
    encoded_text_tokens,
    initial_messages,
    parse_assistant,
    stop_sequences,
    tool_message,
)
from .reward import JudgeResult, invalid_format_result


@dataclass(frozen=True)
class RolloutConfig:
    prompt_variant: str = "c1"
    system_prompt_override: str | None = None
    group_size: int = 4
    max_tool_calls: int = 1
    max_assistant_turns: int = 2
    max_trajectory_tokens: int = 8192
    max_assistant_tokens: int = 2048
    max_tool_response_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int = 42


@dataclass
class AssistantTurn:
    prompt_tokens: list[int]
    completion_tokens: list[int]
    logprobs: list[float]
    text: str
    stop_reason: str = ""
    requested_max_tokens: int = 0
    completion_token_count: int = 0
    hit_token_limit: bool = False


@dataclass
class Trajectory:
    example: LCBExample
    question_index: int
    group_index: int
    messages: list[dict[str, Any]]
    next_prompt_tokens: list[int] | None = None
    next_assistant_prefix: str = ""
    turns: list[AssistantTurn] = field(default_factory=list)
    tool_calls: int = 0
    tool_call_attempts: int = 0
    valid_tool_calls: int = 0
    final_text: str = ""
    final_code: str | None = None
    judge_result: JudgeResult | None = None
    reward: float = 0.0
    advantage: float = 0.0
    duration_seconds: float = 0.0
    started_at: float = field(default_factory=perf_counter)
    trajectory_budget_exhausted: bool = False
    done: bool = False

    @property
    def hit_token_limit(self) -> bool:
        return self.trajectory_budget_exhausted or any(
            turn.hit_token_limit for turn in self.turns
        )

    @property
    def finish_reason(self) -> str:
        if self.hit_token_limit:
            return "max_tokens"
        if self.turns and self.turns[-1].stop_reason:
            return self.turns[-1].stop_reason
        return "agent_final"


@dataclass(frozen=True)
class PendingTool:
    trajectory: Trajectory
    messages_before_assistant: list[dict[str, Any]]
    prompt_tokens: list[int]
    completion_tokens: list[int]
    code: str
    stdin: str
    call_id: str


def _read_sequence(sequence: Any, tokenizer: Any) -> tuple[list[int], list[float], str]:
    tokens = [int(value) for value in sequence.tokens]
    logprobs = [float(value) for value in sequence.logprobs]
    if len(tokens) != len(logprobs):
        raise ValueError("Sample tokens and old logprobs are not aligned")
    text = sequence.text
    if text is None:
        text = tokenizer.decode(tokens, skip_special_tokens=True)
    return tokens, logprobs, str(text).strip()


TOKEN_LIMIT_STOP_REASONS = frozenset(
    {"length", "max_length", "max_tokens", "token_limit", "context_length"}
)


def _stop_reason(sequence: Any) -> str:
    value = getattr(sequence, "stop_reason", None)
    if value is None:
        # Compatibility only for test doubles and older adapters. PyTRIO 0.2.5 uses
        # stop_reason and production artifacts always persist that field.
        value = getattr(sequence, "finish_reason", None)
    return str(value or "").strip().lower()


def _hit_token_limit(stop_reason: str, completion_tokens: int, requested_max_tokens: int) -> bool:
    return (
        stop_reason in TOKEN_LIMIT_STOP_REASONS
        or requested_max_tokens > 0
        and completion_tokens >= requested_max_tokens
    )


def _token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _fit_observation(
    tokenizer: Any,
    result: ToolResult,
    pending: PendingTool,
    config: RolloutConfig,
    final_submission_required: bool = False,
) -> tuple[str, list[int]] | None:
    content = result.observation()
    if final_submission_required:
        content = f"{content}\n\n[Controller instruction]\n{FINAL_SUBMISSION_REMINDER}"
    while True:
        if _token_count(tokenizer, content) <= config.max_tool_response_tokens:
            message = tool_message(pending.call_id, content)
            next_prompt = build_next_prompt(
                tokenizer,
                pending.messages_before_assistant,
                pending.prompt_tokens,
                pending.completion_tokens,
                message,
            )
            if final_submission_required:
                next_prompt = [
                    *next_prompt,
                    *encoded_text_tokens(tokenizer, FINAL_CODE_PREFIX),
                ]
            if len(next_prompt) <= config.max_trajectory_tokens:
                return content, next_prompt
        if len(content) <= 96:
            return None
        keep = max(32, int(len(content) * 0.35))
        content = content[:keep] + "\n[... truncated ...]\n" + content[-keep:]


def _begin_turn(
    trajectory: Trajectory,
    prompt_tokens: list[int],
    sequence: Any,
    tokenizer: Any,
    config: RolloutConfig,
    requested_max_tokens: int,
) -> PendingTool | None:
    tokens, logprobs, sampled_text = _read_sequence(sequence, tokenizer)
    stop_reason = _stop_reason(sequence)
    assistant_prefix = trajectory.next_assistant_prefix
    trajectory.next_assistant_prefix = ""
    if assistant_prefix and "```" not in sampled_text:
        sampled_text = f"{sampled_text.rstrip()}\n```"
    text = f"{assistant_prefix}{sampled_text}"
    trajectory.turns.append(
        AssistantTurn(
            prompt_tokens=prompt_tokens,
            completion_tokens=tokens,
            logprobs=logprobs,
            text=text,
            stop_reason=stop_reason,
            requested_max_tokens=requested_max_tokens,
            completion_token_count=len(tokens),
            hit_token_limit=_hit_token_limit(stop_reason, len(tokens), requested_max_tokens),
        )
    )
    parsed = parse_assistant(text)
    if "<tool_call>" in text:
        trajectory.tool_call_attempts += 1
    if parsed.kind == "tool":
        trajectory.valid_tool_calls += 1
    can_call = (
        parsed.kind == "tool"
        and trajectory.tool_calls < config.max_tool_calls
        and len(trajectory.turns) < config.max_assistant_turns
    )
    if can_call:
        before = list(trajectory.messages)
        trajectory.messages.append({"role": "assistant", "content": text})
        return PendingTool(
            trajectory=trajectory,
            messages_before_assistant=before,
            prompt_tokens=prompt_tokens,
            completion_tokens=tokens,
            code=parsed.code or "",
            stdin=parsed.stdin,
            call_id=(
                f"lcb-q{trajectory.question_index}-g{trajectory.group_index}-"
                f"t{trajectory.tool_calls + 1}"
            ),
        )
    trajectory.messages.append({"role": "assistant", "content": text})
    trajectory.final_text = text
    trajectory.final_code = parsed.code if parsed.kind == "final" else None
    trajectory.done = True
    return None


async def _sample(
    sampling_client: Any,
    prompt_tokens: list[int],
    num_samples: int,
    seed: int,
    max_tokens: int,
    config: RolloutConfig,
    tokenizer: Any,
    final_submission: bool = False,
) -> Any:
    return await sampling_client.sample_async(
        prompt=trio.ModelInput.from_ints(prompt_tokens),
        num_samples=num_samples,
        sampling_params=trio.SamplingParams(
            max_tokens=max_tokens,
            seed=seed,
            stop=stop_sequences(tokenizer, final_submission=final_submission),
            temperature=config.temperature,
            top_p=config.top_p,
        ),
        return_text=True,
    )


async def _execute_pending(
    pending: list[PendingTool],
    executor: DockerExecutor,
    tokenizer: Any,
    config: RolloutConfig,
) -> None:
    results = await asyncio.gather(
        *(executor.arun_tool(item.code, item.stdin) for item in pending)
    )
    for item, result in zip(pending, results, strict=True):
        trajectory = item.trajectory
        trajectory.tool_calls += 1
        must_submit = (
            trajectory.tool_calls >= config.max_tool_calls
            or len(trajectory.turns) >= config.max_assistant_turns - 1
        )
        fitted = _fit_observation(
            tokenizer,
            result,
            item,
            config,
            final_submission_required=must_submit,
        )
        if fitted is None:
            trajectory.final_text = trajectory.turns[-1].text
            trajectory.done = True
            continue
        content, next_prompt = fitted
        trajectory.messages.append(tool_message(item.call_id, content))
        trajectory.next_prompt_tokens = next_prompt
        trajectory.next_assistant_prefix = FINAL_CODE_PREFIX if must_submit else ""


async def rollout_batch_async(
    sampling_client: Any,
    tokenizer: Any,
    executor: DockerExecutor,
    examples: list[LCBExample],
    config: RolloutConfig,
    progress_callback: Callable[[int], None] | None = None,
) -> list[Trajectory]:
    """Sample, execute tools, judge code, and score same-prompt rollout groups.

    First turns are sampled as one PyTRIO group per question. Later turns are
    sampled individually after sandboxed tool observations are appended. The
    returned trajectories preserve rollout-time token logprobs for GRPO.
    """

    if config.group_size < 1:
        raise ValueError("group_size must be positive")
    if config.max_tool_calls < 0:
        raise ValueError("max_tool_calls must be non-negative")
    trajectories = [
        Trajectory(
            example=example,
            question_index=question_index,
            group_index=group_index,
            messages=initial_messages(
                example,
                config.prompt_variant,
                max_tool_calls=config.max_tool_calls,
                override_system_prompt=config.system_prompt_override,
            ),
        )
        for question_index, example in enumerate(examples)
        for group_index in range(config.group_size)
    ]

    first_calls = []
    for question_index, example in enumerate(examples):
        representative = trajectories[question_index * config.group_size]
        prompt = build_prompt(tokenizer, representative.messages)
        max_tokens = min(config.max_assistant_tokens, config.max_trajectory_tokens - len(prompt))
        if max_tokens <= 0:
            raise ValueError(f"Initial prompt exceeds trajectory budget for {example.instance_id}")
        first_calls.append(
            _sample(
                sampling_client,
                prompt,
                config.group_size,
                config.seed + question_index * 100,
                max_tokens,
                config,
                tokenizer,
            )
        )
    first_results = await asyncio.gather(*first_calls)
    pending: list[PendingTool] = []
    for question_index, response in enumerate(first_results):
        if len(response.sequences) != config.group_size:
            raise RuntimeError("Sampler returned the wrong number of group sequences")
        for group_index, sequence in enumerate(response.sequences):
            trajectory = trajectories[question_index * config.group_size + group_index]
            prompt = build_prompt(tokenizer, trajectory.messages)
            item = _begin_turn(
                trajectory,
                prompt,
                sequence,
                tokenizer,
                config,
                requested_max_tokens=min(
                    config.max_assistant_tokens,
                    config.max_trajectory_tokens - len(prompt),
                ),
            )
            if item is not None:
                pending.append(item)
    await _execute_pending(pending, executor, tokenizer, config)

    while True:
        active = [item for item in trajectories if not item.done]
        if not active:
            break
        requests = []
        request_meta: list[tuple[Trajectory, list[int], int]] = []
        for trajectory in active:
            prompt = trajectory.next_prompt_tokens or build_prompt(tokenizer, trajectory.messages)
            max_tokens = min(config.max_assistant_tokens, config.max_trajectory_tokens - len(prompt))
            if max_tokens <= 0:
                trajectory.trajectory_budget_exhausted = True
                trajectory.done = True
                continue
            request_meta.append((trajectory, prompt, max_tokens))
            requests.append(
                _sample(
                    sampling_client,
                    prompt,
                    1,
                    config.seed
                    + trajectory.question_index * 100
                    + trajectory.group_index * 10
                    + len(trajectory.turns),
                    max_tokens,
                    config,
                    tokenizer,
                    final_submission=bool(trajectory.next_assistant_prefix),
                )
            )
        if not requests:
            break
        responses = await asyncio.gather(*requests)
        pending = []
        for (trajectory, prompt, requested_max_tokens), response in zip(
            request_meta, responses, strict=True
        ):
            item = _begin_turn(
                trajectory,
                prompt,
                response.sequences[0],
                tokenizer,
                config,
                requested_max_tokens,
            )
            if item is not None:
                pending.append(item)
        await _execute_pending(pending, executor, tokenizer, config)

    judge_tasks = []
    judge_trajectories = []
    for trajectory in trajectories:
        if trajectory.final_code is None:
            trajectory.judge_result = invalid_format_result(
                total=len(trajectory.example.all_tests),
                public_total=len(trajectory.example.public_tests),
                private_total=len(trajectory.example.private_tests),
            )
        else:
            judge_trajectories.append(trajectory)
            judge_tasks.append(executor.ajudge(trajectory.example, trajectory.final_code))
    if judge_tasks:
        results = await asyncio.gather(*judge_tasks)
        for trajectory, result in zip(judge_trajectories, results, strict=True):
            trajectory.judge_result = result

    for trajectory in trajectories:
        assert trajectory.judge_result is not None
        trajectory.reward = trajectory.judge_result.reward
        trajectory.duration_seconds = perf_counter() - trajectory.started_at
        if progress_callback:
            progress_callback(1)
    assign_group_advantages(trajectories)
    return trajectories


def assign_group_advantages(trajectories: list[Trajectory]) -> None:
    """Set each trajectory's advantage to reward minus its question-group mean."""

    grouped: dict[int, list[Trajectory]] = {}
    for trajectory in trajectories:
        grouped.setdefault(trajectory.question_index, []).append(trajectory)
    for group in grouped.values():
        mean_reward = sum(item.reward for item in group) / len(group)
        for trajectory in group:
            trajectory.advantage = trajectory.reward - mean_reward


def rollout_batch(*args: Any, **kwargs: Any) -> list[Trajectory]:
    """Synchronous wrapper for the async rollout state machine."""

    try:
        return asyncio.run(rollout_batch_async(*args, **kwargs))
    except DockerInfrastructureError:
        raise
