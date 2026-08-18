"""Stateless Docker execution for run_python calls and hidden-test judging."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .data import LCBExample
from .reward import JudgeResult


DEFAULT_IMAGE = "retool-coding-0812-sandbox:py311"
MAX_CAPTURE_BYTES = 128 * 1024


class DockerInfrastructureError(RuntimeError):
    retryable = True


@dataclass(frozen=True)
class DockerConfig:
    docker_binary: str = "docker"
    image: str = DEFAULT_IMAGE
    max_workers: int = 4
    cpus: float = 1.0
    memory: str = "512m"
    pids_limit: int = 64
    tmpfs_size: str = "64m"
    tool_timeout_seconds: float = 5.0
    case_timeout_seconds: float = 3.0
    judge_timeout_seconds: float = 30.0


@dataclass(frozen=True)
class ToolResult:
    status: str
    stdout: str
    stderr: str
    execution_seconds: float

    def observation(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class DockerStats:
    calls: int = 0
    tool_calls: int = 0
    judge_calls: int = 0
    infrastructure_errors: int = 0
    timeouts: int = 0
    seconds: float = 0.0

    def metrics(self) -> dict[str, float]:
        total = max(self.calls, 1)
        return {
            "docker/calls": float(self.calls),
            "docker/tool_calls": float(self.tool_calls),
            "docker/judge_calls": float(self.judge_calls),
            "docker/infrastructure_error_rate": self.infrastructure_errors / total,
            "docker/timeout_rate": self.timeouts / total,
            "docker/mean_seconds": self.seconds / total,
        }

    def snapshot(self) -> dict[str, float]:
        return {
            key: float(value)
            for key, value in asdict(self).items()
        }

    def metrics_since(self, previous: dict[str, float]) -> dict[str, float]:
        current = self.snapshot()
        delta = {key: current[key] - float(previous.get(key, 0.0)) for key in current}
        total = max(delta["calls"], 1.0)
        return {
            "docker/calls": delta["calls"],
            "docker/tool_calls": delta["tool_calls"],
            "docker/judge_calls": delta["judge_calls"],
            "docker/infrastructure_error_rate": delta["infrastructure_errors"] / total,
            "docker/timeout_rate": delta["timeouts"] / total,
            "docker/mean_seconds": delta["seconds"] / total,
        }


class DockerExecutor:
    def __init__(self, config: DockerConfig = DockerConfig()) -> None:
        if not 1 <= config.max_workers <= 4:
            raise ValueError("max_workers must be between 1 and the experiment limit of 4")
        self.config = config
        self.stats = DockerStats()
        self._semaphore = threading.BoundedSemaphore(config.max_workers)

    def docker_args(self) -> list[str]:
        config = self.config
        return [
            config.docker_binary,
            "run",
            "--rm",
            "-i",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(config.pids_limit),
            "--memory",
            config.memory,
            "--memory-swap",
            config.memory,
            "--cpus",
            str(config.cpus),
            "--tmpfs",
            f"/tmp:rw,noexec,nosuid,size={config.tmpfs_size}",
            "--ulimit",
            "nofile=64:64",
            "--user",
            "65532:65532",
            config.image,
        ]

    @staticmethod
    def _read_capped(stream: Any) -> bytes:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - MAX_CAPTURE_BYTES))
        return stream.read()

    def _run_payload(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        started = time.perf_counter()
        self.stats.calls += 1
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        with self._semaphore:
            with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
                try:
                    process = subprocess.Popen(
                        self.docker_args(),
                        stdin=subprocess.PIPE,
                        stdout=stdout_file,
                        stderr=stderr_file,
                    )
                except OSError as exc:
                    self.stats.infrastructure_errors += 1
                    raise DockerInfrastructureError(f"Cannot start Docker: {exc}") from exc
                try:
                    process.communicate(input=encoded, timeout=timeout)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    result = self._execution_failure_result(
                        payload,
                        status="time_limit",
                        detail=f"sandbox exceeded {timeout:.1f}s",
                    )
                    self.stats.seconds += time.perf_counter() - started
                    return result
                stdout = self._read_capped(stdout_file).decode("utf-8", errors="replace")
                stderr = self._read_capped(stderr_file).decode("utf-8", errors="replace")
        self.stats.seconds += time.perf_counter() - started
        if process.returncode in {-9, 137}:
            return self._execution_failure_result(
                payload,
                status="runtime_error",
                detail="sandbox was killed (likely memory/resource limit)",
            )
        if process.returncode != 0:
            self.stats.infrastructure_errors += 1
            raise DockerInfrastructureError(
                f"Docker exited {process.returncode}: {stderr[-2000:]}"
            )
        try:
            result = json.loads(stdout)
        except json.JSONDecodeError as exc:
            self.stats.infrastructure_errors += 1
            raise DockerInfrastructureError(
                f"Sandbox returned invalid JSON: {stdout[-2000:]}"
            ) from exc
        if result.get("worker_error"):
            self.stats.infrastructure_errors += 1
            raise DockerInfrastructureError(str(result["worker_error"]))
        return dict(result)

    @staticmethod
    def _execution_failure_result(
        payload: dict[str, Any], *, status: str, detail: str
    ) -> dict[str, Any]:
        if payload.get("mode") == "tool":
            return {
                "status": status,
                "stdout": "",
                "stderr": detail,
                "execution_seconds": 0.0,
            }
        tests = list(payload.get("tests") or [])
        return {
            "passed": 0,
            "total": len(tests),
            "public_passed": 0,
            "public_total": sum(item.get("source") == "public" for item in tests),
            "private_passed": 0,
            "private_total": sum(item.get("source") == "private" for item in tests),
            "status": status,
            "first_failure": detail,
            "execution_seconds": 0.0,
        }

    def run_tool(self, code: str, stdin: str = "") -> ToolResult:
        self.stats.tool_calls += 1
        result = self._run_payload(
            {
                "mode": "tool",
                "code": code,
                "stdin": stdin,
                "timeout_seconds": self.config.tool_timeout_seconds,
            },
            timeout=self.config.tool_timeout_seconds + 5.0,
        )
        output = ToolResult(
            status=str(result.get("status", "runtime_error")),
            stdout=str(result.get("stdout", "")),
            stderr=str(result.get("stderr", "")),
            execution_seconds=float(result.get("execution_seconds", 0.0)),
        )
        if output.status == "time_limit":
            self.stats.timeouts += 1
        return output

    async def arun_tool(self, code: str, stdin: str = "") -> ToolResult:
        return await asyncio.to_thread(self.run_tool, code, stdin)

    def judge(self, example: LCBExample, code: str) -> JudgeResult:
        self.stats.judge_calls += 1
        tests = [
            {**asdict(case), "source": source}
            for source, cases in (("public", example.public_tests), ("private", example.private_tests))
            for case in cases
        ]
        result = self._run_payload(
            {
                "mode": "judge",
                "code": code,
                "fn_name": example.fn_name,
                "tests": tests,
                "case_timeout_seconds": self.config.case_timeout_seconds,
                "judge_timeout_seconds": max(self.config.judge_timeout_seconds - 1.0, 0.1),
            },
            timeout=self.config.judge_timeout_seconds,
        )
        output = JudgeResult(
            passed=int(result.get("passed", 0)),
            total=int(result.get("total", len(tests))),
            public_passed=int(result.get("public_passed", 0)),
            public_total=int(result.get("public_total", len(example.public_tests))),
            private_passed=int(result.get("private_passed", 0)),
            private_total=int(result.get("private_total", len(example.private_tests))),
            status=str(result.get("status", "runtime_error")),
            first_failure=str(result.get("first_failure", "")),
            execution_seconds=float(result.get("execution_seconds", 0.0)),
        )
        if output.status == "time_limit":
            self.stats.timeouts += 1
        return output

    async def ajudge(self, example: LCBExample, code: str) -> JudgeResult:
        return await asyncio.to_thread(self.judge, example, code)

    def check_runtime(self) -> str:
        command = [self.config.docker_binary, "version", "--format", "{{.Server.Version}}"]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
        except OSError as exc:
            raise DockerInfrastructureError(f"Docker CLI is unavailable: {exc}") from exc
        if result.returncode != 0 or not result.stdout.strip():
            raise DockerInfrastructureError(
                f"Docker daemon is unavailable: {(result.stderr or result.stdout).strip()}"
            )
        inspect = subprocess.run(
            [
                self.config.docker_binary,
                "image",
                "inspect",
                self.config.image,
                "--format",
                "{{.Id}} {{.Architecture}}",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if inspect.returncode != 0:
            raise DockerInfrastructureError(
                f"Sandbox image {self.config.image!r} is missing; run preflight.py"
            )
        fields = inspect.stdout.strip().split()
        if len(fields) != 2:
            raise DockerInfrastructureError("Docker returned malformed image metadata")
        image_id, architecture = fields
        if architecture != "arm64":
            raise DockerInfrastructureError(
                f"Sandbox image must be native arm64, got {architecture!r}"
            )
        return image_id


def build_image(
    project_root: str | Path,
    *,
    docker_binary: str = "docker",
    image: str = DEFAULT_IMAGE,
) -> str:
    docker_dir = Path(project_root) / "docker"
    try:
        daemon = subprocess.run(
            [docker_binary, "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except OSError as exc:
        raise DockerInfrastructureError(f"Docker CLI is unavailable: {exc}") from exc
    if daemon.returncode != 0 or not daemon.stdout.strip():
        raise DockerInfrastructureError(
            f"Docker daemon is unavailable: {(daemon.stderr or daemon.stdout).strip()}"
        )
    result = subprocess.run(
        [
            docker_binary,
            "build",
            "--pull",
            "--platform",
            "linux/arm64",
            "--tag",
            image,
            str(docker_dir),
        ],
        check=False,
    )
    if result.returncode != 0:
        raise DockerInfrastructureError("Failed to build the sandbox image")
    executor = DockerExecutor(DockerConfig(docker_binary=docker_binary, image=image))
    return executor.check_runtime()
