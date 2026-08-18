"""MyEval benchmark and agent executor for ReTool-Coding-0812."""

from __future__ import annotations

from dataclasses import asdict
import asyncio
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from myeval.benchmarks.base import BenchmarkPlugin, DatasetSnapshot
from myeval.execution import TaskExecutor
from myeval.fingerprint import file_sha256, fingerprint
from myeval.registry import BENCHMARKS, EXECUTORS
from myeval.types import EvalSample, GenerationRequest, GenerationResult, SampleScore

from .data import LCBExample, load_examples
from .docker_executor import DEFAULT_IMAGE, DockerConfig, DockerExecutor
from .protocol import user_prompt
from .rollout import RolloutConfig, rollout_batch_async
from .telemetry import aggregate_behavior_metrics, save_trajectory_gzip


class LCBOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    docker_binary: str = "docker"
    image: str = DEFAULT_IMAGE
    max_tool_calls: int = Field(default=1, ge=1)
    max_assistant_turns: int = Field(default=2, ge=1)
    max_trajectory_tokens: int = Field(default=20480, ge=256)
    max_tool_response_tokens: int = Field(default=512, ge=64)
    sandbox_cpus: float = Field(default=1.0, gt=0)
    sandbox_memory: str = "512m"
    tool_timeout_seconds: float = Field(default=5.0, gt=0)
    case_timeout_seconds: float = Field(default=3.0, gt=0)
    judge_timeout_seconds: float = Field(default=30.0, gt=0)


LCB_METRICS = (
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


@BENCHMARKS.register("lcb_codegen_retool_0812")
class LCBCodegenMini(BenchmarkPlugin):
    """Frozen LiveCodeBench snapshot whose scoring requires the agent executor."""

    id = "lcb_codegen_retool_0812"
    version = "1"
    executor_id = "lcb_retool_agent_0812"
    primary_metric = "pass_at_1"
    report_metrics = LCB_METRICS
    failure_metrics = {metric: 0.0 for metric in LCB_METRICS}
    expose_reference_in_report = False

    def _load_examples_by_id(self) -> dict[str, LCBExample]:
        if self.config.local_path is None:
            raise ValueError("lcb_codegen_retool_0812 requires benchmark.local_path")
        examples = load_examples(self.config.local_path)
        return {example.instance_id: example for example in examples}

    def example(self, sample_id: str) -> LCBExample:
        examples = getattr(self, "_examples_by_id", None)
        if examples is None:
            examples = self._load_examples_by_id()
            self._examples_by_id = examples
        try:
            return examples[sample_id]
        except KeyError as exc:
            raise ValueError(f"Unknown frozen LCB sample: {sample_id}") from exc

    def implementation_fingerprint(self) -> str:
        source_root = Path(__file__).resolve().parent
        return fingerprint(
            {
                "id": self.id,
                "version": self.version,
                "files": {
                    str(path.relative_to(source_root)): file_sha256(path)
                    for path in sorted(source_root.glob("*.py"))
                },
            }
        )

    def validate_options(self, options: dict[str, Any]) -> None:
        self.options = LCBOptions.model_validate(options)

    def load(self) -> DatasetSnapshot:
        examples_by_id = self._load_examples_by_id()
        self._examples_by_id = examples_by_id
        examples = list(examples_by_id.values())
        source = {
            "kind": "lcb_frozen_file",
            "path": str(self.config.local_path.resolve()),
            "sha256": file_sha256(self.config.local_path),
            "split": self.config.split,
        }
        samples = []
        for example in examples:
            samples.append(
                EvalSample(
                    id=example.instance_id,
                    input=example.question_content,
                    # Hidden tests remain in the frozen dataset and are resolved by
                    # sample id inside the executor. Persisting them here duplicates
                    # large private-test payloads for every model/prompt task.
                    reference={"instance_id": example.instance_id},
                    metadata={
                        "platform": example.platform,
                        "difficulty": example.difficulty,
                        "contest_date": example.contest_date,
                        "test_type": example.test_type,
                    },
                )
            )
        return self.finalize_snapshot(samples, [], source)

    def build_user_prompt(self, sample: EvalSample) -> str:
        return user_prompt(self.example(sample.id))

    def score(self, sample: EvalSample, prediction: str) -> SampleScore:
        del sample, prediction
        raise RuntimeError("lcb_codegen_retool_0812 requires its structured agent executor")

    def score_result(self, sample: EvalSample, result: GenerationResult) -> SampleScore:
        del sample
        judge = dict(result.metadata.get("judge_result") or {})
        behavior = dict(result.metadata.get("behavior") or {})
        total = int(judge.get("total", 0))
        public_total = int(judge.get("public_total", 0))
        private_total = int(judge.get("private_total", 0))
        status = str(judge.get("status", "infrastructure_error"))
        metrics = {
            "pass_at_1": float(bool(judge.get("resolved", False))),
            "case_pass_rate": int(judge.get("passed", 0)) / max(total, 1),
            "public_pass_rate": int(judge.get("public_passed", 0)) / max(public_total, 1),
            "private_pass_rate": int(judge.get("private_passed", 0)) / max(private_total, 1),
            "format_valid_rate": float(status != "invalid_format"),
            "token_cap_hit_rate": float(behavior.get("token_cap_hit_rate", 0.0)),
            "compile_error_rate": float(status == "compile_error"),
            "runtime_error_rate": float(status == "runtime_error"),
            "time_limit_rate": float(status == "time_limit"),
        }
        for key in (
            "tool_use_rate",
            "mean_tool_calls",
            "tool_call_valid_rate",
            "mean_turns",
            "mean_trajectory_tokens",
            "mean_execution_seconds",
        ):
            metrics[key] = float(behavior.get(key, 0.0))
        return SampleScore(
            metrics=metrics,
            parsed_answer={
                "artifact_path": result.metadata.get("artifact_path"),
                "status": status,
            },
            grader="execution",
            details={
                "artifact_path": result.metadata.get("artifact_path"),
                "judge_result": judge,
            },
        )


@EXECUTORS.register("lcb_retool_agent_0812")
class LCBReToolExecutor(TaskExecutor):
    """Run the ReTool agent loop and judge its final code in the Docker sandbox."""

    def __init__(self, plugin: BenchmarkPlugin, config: Any, run_dir: Path) -> None:
        super().__init__(plugin, config, run_dir)
        assert isinstance(plugin, LCBCodegenMini)
        self.lcb_plugin = plugin
        options = plugin.options
        self.options = options
        self.executor = DockerExecutor(
            DockerConfig(
                docker_binary=options.docker_binary,
                image=options.image,
                max_workers=config.execution.max_concurrency,
                cpus=options.sandbox_cpus,
                memory=options.sandbox_memory,
                tool_timeout_seconds=options.tool_timeout_seconds,
                case_timeout_seconds=options.case_timeout_seconds,
                judge_timeout_seconds=options.judge_timeout_seconds,
            )
        )

    async def initialize(self) -> None:
        # Fail before a PyTRIO model is provisioned when local execution is absent.
        await asyncio.to_thread(self.executor.check_runtime)

    async def execute(
        self,
        request: GenerationRequest,
        sample: EvalSample,
        adapter: Any,
    ) -> GenerationResult:
        if adapter.config.backend != "pytrio":
            raise ValueError("lcb_codegen_retool_0812 supports only the pytrio backend")
        runtime = await adapter.token_sampling_runtime()
        example = self.lcb_plugin.example(sample.id)
        system = next(
            (
                message.content
                for message in request.messages or ()
                if message.role == "system"
            ),
            None,
        )
        trajectories = await rollout_batch_async(
            runtime.sampling_client,
            runtime.tokenizer,
            self.executor,
            [example],
            RolloutConfig(
                prompt_variant="c0",
                system_prompt_override=system,
                group_size=1,
                max_tool_calls=self.options.max_tool_calls,
                max_assistant_turns=self.options.max_assistant_turns,
                max_trajectory_tokens=self.options.max_trajectory_tokens,
                max_assistant_tokens=request.params.max_out_length,
                max_tool_response_tokens=self.options.max_tool_response_tokens,
                temperature=request.params.temperature,
                top_p=request.params.top_p,
                seed=request.params.seed or self.config.experiment.seed,
            ),
        )
        trajectory = trajectories[0]
        artifact = self.artifact_dir / f"{request.task_key}.json.gz"
        save_trajectory_gzip(
            artifact,
            trajectory,
            metadata={
                "task_key": request.task_key,
                "model_id": request.model_id,
                "system_prompt_id": request.condition.get("system_prompt_id"),
            },
        )
        if trajectory.judge_result is None:
            raise RuntimeError("agent rollout completed without a judge result")
        judge = asdict(trajectory.judge_result)
        judge["resolved"] = bool(trajectory.judge_result.resolved)
        judge["reward"] = float(trajectory.judge_result.reward)
        return GenerationResult(
            text=trajectory.final_text,
            prompt_tokens=len(trajectory.turns[0].prompt_tokens) if trajectory.turns else 0,
            completion_tokens=sum(len(turn.completion_tokens) for turn in trajectory.turns),
            finish_reason=trajectory.finish_reason,
            latency_seconds=trajectory.duration_seconds,
            deterministic=request.params.temperature == 0,
            metadata={
                "artifact_path": str(artifact.relative_to(self.run_dir)),
                "judge_result": judge,
                "behavior": aggregate_behavior_metrics(trajectories),
            },
        )


def register() -> None:
    """Entry-point import performs registration."""
