"""Launch and cheaply supervise the frozen 01_train pipeline."""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from datetime import UTC, datetime
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Iterator

from retool_coding_0812.resume import latest_checkpoint, load_checkpoint, read_metric_rows
from retool_coding_0812.settings import ROOT, load_config


TRAINING_ROOT = ROOT / "artifacts/training"
SUPERVISOR_DIR = TRAINING_ROOT / "supervisor"
STATUS_PATH = TRAINING_ROOT / "supervisor-status.json"
LOCK_PATH = SUPERVISOR_DIR / "supervisor.lock"
RUN_LOG = SUPERVISOR_DIR / "run-all.log"
INTERNAL_LOG = SUPERVISOR_DIR / "supervisor.log"
REPORT_PATH = ROOT / "docs/experiment-record-and-results.md"
MILESTONE_STEPS = (20, 40, 60, 80, 100)
EXPECTED_TRAJECTORIES = 32
MAX_RECOVERIES = 3

RETRYABLE_PATTERNS = {
    "http_transient": re.compile(r"\b(?:408|429|500|502|503|504)\b|rate.?limit", re.I),
    "network": re.compile(
        r"connection(?:error|reset|aborted)?|timed?\s*out|temporary failure|"
        r"remote disconnected|server disconnected|network is unreachable",
        re.I,
    ),
    "docker_transient": re.compile(
        r"Docker daemon is unavailable|Cannot connect to the Docker daemon|"
        r"docker.*(?:timed?\s*out|connection)",
        re.I,
    ),
}
FATAL_PATTERNS = {
    "billing_insufficient_balance": re.compile(r"billing_insufficient_balance", re.I),
    "early_gate_failed": re.compile(r"Early training signal gate failed|early.*gate.*failed", re.I),
    "authentication": re.compile(r"\b(?:401|403)\b|unauthorized|forbidden|authentication", re.I),
    "state_inconsistent": re.compile(
        r"invariant|fingerprint|contiguous|authoritative|checkpoint.*(?:invalid|missing)|"
        r"manifest.*(?:invalid|mismatch)|trajectory.*exactly",
        re.I,
    ),
    "configuration": re.compile(
        r"ValueError|FileNotFoundError|ModuleNotFoundError|configuration drifted|"
        r"data.*(?:mismatch|overrun)",
        re.I,
    ),
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _default_status() -> dict[str, Any]:
    return {
        "version": 1,
        "pipeline": "01_train",
        "state": "idle",
        "stage": "shared-sft",
        "runner": {
            "supervisor_pid": None,
            "pid": None,
            "started_at": None,
            "alive": False,
            "exit_code": None,
            "command": None,
        },
        "progress": {
            "step": 0,
            "checkpoint_step": 0,
            "checkpoint_path": None,
            "swanlab_run_id": None,
            "early_gate": None,
            "branches": {},
        },
        "log_cursors": {str(RUN_LOG): 0},
        "last_event": None,
        "last_event_fingerprint": None,
        "last_reported_event_fingerprint": None,
        "observed_milestones": [],
        "error": None,
        "recovery": {"error_key": None, "attempts": 0, "max_attempts": MAX_RECOVERIES},
        "last_checked_at": None,
        "updated_at": _now(),
    }


def _load_status() -> dict[str, Any]:
    if not STATUS_PATH.exists():
        return _default_status()
    value = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or int(value.get("version", 0)) != 1:
        raise ValueError(f"Invalid supervisor status: {STATUS_PATH}")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _save_status(status: dict[str, Any]) -> None:
    status["updated_at"] = _now()
    _write_json_atomic(STATUS_PATH, status)


@contextmanager
def _lock() -> Iterator[None]:
    SUPERVISOR_DIR.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid < 1:
        return False
    try:
        os.kill(pid, 0)
    except PermissionError as exc:
        return exc.errno == errno.EPERM
    except ProcessLookupError:
        return False
    except OSError as exc:
        return exc.errno == errno.EPERM
    return True


def _read_new_log(status: dict[str, Any]) -> tuple[str, int]:
    cursor = int(status.setdefault("log_cursors", {}).get(str(RUN_LOG), 0))
    if not RUN_LOG.exists():
        return "", cursor
    size = RUN_LOG.stat().st_size
    if cursor > size:
        cursor = 0
    with RUN_LOG.open("rb") as stream:
        stream.seek(cursor)
        payload = stream.read()
        position = stream.tell()
    return payload.decode("utf-8", errors="replace"), position


def _tail(path: Path, limit: int = 24_000) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - limit))
        return stream.read().decode("utf-8", errors="replace")


def _classify_error(text: str) -> dict[str, Any]:
    for key, pattern in FATAL_PATTERNS.items():
        if pattern.search(text):
            return {"key": key, "retryable": False, "observed_at": _now()}
    for key, pattern in RETRYABLE_PATTERNS.items():
        if pattern.search(text):
            return {"key": key, "retryable": True, "observed_at": _now()}
    return {"key": "unknown_failure", "retryable": False, "observed_at": _now()}


def _metrics(variant: str) -> list[dict[str, Any]]:
    return read_metric_rows(
        TRAINING_ROOT
        / "runs"
        / f"retool-coding-0812-{variant}-seed42"
        / "metrics.jsonl"
    )


def _checkpoint(variant: str) -> tuple[int, Path | None, dict[str, Any] | None]:
    run_name = f"retool-coding-0812-{variant}-seed42"
    path = latest_checkpoint(
        TRAINING_ROOT / "checkpoints",
        pattern=f"{run_name}-step-*.json",
        expected_run_name=run_name,
    )
    if path is None:
        return 0, None, None
    record = load_checkpoint(path)
    return int(record["step"]), path, record


def _validate_branch(variant: str, rows: list[dict[str, Any]]) -> None:
    steps = [int(row["step"]) for row in rows]
    if steps != list(range(1, len(rows) + 1)) or len(rows) > 100:
        raise ValueError(f"{variant} metrics are not contiguous through step {len(rows)}")
    run_name = f"retool-coding-0812-{variant}-seed42"
    for step in MILESTONE_STEPS:
        path = TRAINING_ROOT / "checkpoints" / f"{run_name}-step-{step}.json"
        if step > len(rows):
            if path.exists():
                raise ValueError(f"{variant} checkpoint {step} is beyond local metrics")
            continue
        if step == 100 or step < len(rows) or len(rows) % 20 == 0:
            if not path.exists():
                raise ValueError(f"{variant} checkpoint {step} is missing")
            record = load_checkpoint(path)
            if not record.get("sampler_weights_path"):
                raise ValueError(f"{variant} checkpoint {step} lacks sampler weights")
            trajectory_dir = (
                TRAINING_ROOT / "runs" / run_name / "trajectories" / f"step-{step:04d}"
            )
            if len(list(trajectory_dir.glob("*.json.gz"))) != EXPECTED_TRAJECTORIES:
                raise ValueError(f"{variant} checkpoint {step} lacks authoritative trajectories")


def _snapshot() -> dict[str, Any]:
    config = load_config()
    expected_sft_steps = 0
    sft_metrics = read_metric_rows(
        TRAINING_ROOT / "shared-sft/seed42/metrics.jsonl"
    )
    sft_manifest_path = TRAINING_ROOT / "shared-sft/seed42/manifest.json"
    sft_manifest = None
    if sft_manifest_path.exists():
        sft_manifest = json.loads(sft_manifest_path.read_text(encoding="utf-8"))
        if not isinstance(sft_manifest, dict) or not all(
            sft_manifest.get(key)
            for key in ("complete", "state_path", "sampler_weights_path", "swanlab_run_id")
        ):
            raise ValueError("Shared-SFT completion manifest is incomplete")
        expected_sft_steps = int(sft_manifest["optimizer_steps"])
        if len(sft_metrics) != expected_sft_steps:
            raise ValueError("Shared-SFT metrics do not match completion manifest")
    elif sft_metrics:
        steps = [int(row["step"]) for row in sft_metrics]
        if steps != list(range(1, len(steps) + 1)):
            raise ValueError("Shared-SFT metrics are not contiguous")

    branches: dict[str, Any] = {}
    for variant in config["grpo"]["variants"]:
        rows = _metrics(variant)
        _validate_branch(variant, rows)
        checkpoint_step, checkpoint_path, checkpoint_record = _checkpoint(variant)
        gate_path = (
            TRAINING_ROOT
            / "runs"
            / f"retool-coding-0812-{variant}-seed42"
            / "early-gate.json"
        )
        gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.exists() else None
        if gate is not None and not gate.get("passed"):
            raise RuntimeError(f"{variant} early gate failed")
        branches[variant] = {
            "step": len(rows),
            "checkpoint_step": checkpoint_step,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "checkpoint": checkpoint_record,
            "early_gate": gate,
        }

    milestones: list[dict[str, Any]] = []
    sft_checkpoint_dir = TRAINING_ROOT / "shared-sft/seed42/checkpoints"
    for path in sorted(sft_checkpoint_dir.glob("step-*.json")):
        record = load_checkpoint(path)
        step = int(record["step"])
        if step % 20 == 0:
            milestones.append(
                {
                    "id": f"shared-sft:checkpoint:{step}",
                    "kind": "checkpoint",
                    "stage": "shared-sft",
                    "step": step,
                    "path": str(path),
                }
            )
    if sft_manifest is not None:
        milestones.append(
            {
                "id": "shared-sft:complete",
                "kind": "stage_complete",
                "stage": "shared-sft",
                "step": expected_sft_steps,
                "swanlab_run_id": sft_manifest.get("swanlab_run_id"),
            }
        )
    for variant, item in branches.items():
        if item["early_gate"] is not None:
            milestones.append(
                {
                    "id": f"{variant}:early-gate",
                    "kind": "early_gate",
                    "stage": variant,
                    "step": 20,
                    "gate": item["early_gate"],
                }
            )
        for step in MILESTONE_STEPS:
            checkpoint_path = (
                TRAINING_ROOT
                / "checkpoints"
                / f"retool-coding-0812-{variant}-seed42-step-{step}.json"
            )
            if checkpoint_path.exists():
                milestones.append(
                    {
                        "id": f"{variant}:checkpoint:{step}",
                        "kind": "checkpoint",
                        "stage": variant,
                        "step": step,
                        "path": str(checkpoint_path),
                    }
                )

    if sft_manifest is None:
        return {
            "stage": "shared-sft",
            "step": len(sft_metrics),
            "checkpoint_step": max(
                (
                    int(load_checkpoint(path)["step"])
                    for path in (TRAINING_ROOT / "shared-sft/seed42/checkpoints").glob("step-*.json")
                ),
                default=0,
            ),
            "checkpoint_path": None,
            "swanlab_run_id": None,
            "early_gate": None,
            "branches": branches,
            "milestones": milestones,
            "complete": False,
        }
    public_branches = {
        variant: {
            key: value
            for key, value in item.items()
            if key != "checkpoint"
        }
        for variant, item in branches.items()
    }
    max_steps = int(config["grpo"]["steps"])
    if any(int(item["step"]) < max_steps for item in branches.values()):
        return {
            "stage": "grpo-parallel",
            "step": min(int(item["step"]) for item in branches.values()),
            "checkpoint_step": min(
                int(item["checkpoint_step"]) for item in branches.values()
            ),
            "checkpoint_path": None,
            "swanlab_run_id": None,
            "early_gate": None,
            "branches": public_branches,
            "milestones": milestones,
            "complete": False,
        }
    return {
        "stage": "complete",
        "step": 100,
        "checkpoint_step": 100,
        "checkpoint_path": branches["c2"]["checkpoint_path"],
        "swanlab_run_id": (branches["c2"]["checkpoint"] or {}).get("swanlab_run_id"),
        "early_gate": branches["c2"]["early_gate"],
        "branches": public_branches,
        "milestones": milestones,
        "complete": True,
    }


def _event(previous: dict[str, Any], snapshot: dict[str, Any], error: Any) -> dict[str, Any] | None:
    if error is not None:
        return {"kind": "error", "stage": snapshot["stage"], "error": error}
    old = previous.get("progress") or {}
    if snapshot["complete"] and previous.get("state") != "complete":
        return {"kind": "complete", "stage": "complete", "step": 100}
    if snapshot["stage"] != previous.get("stage"):
        return {"kind": "stage_changed", "stage": snapshot["stage"], "step": snapshot["step"]}
    if snapshot.get("early_gate") and not old.get("early_gate"):
        return {"kind": "early_gate", "stage": snapshot["stage"], "gate": snapshot["early_gate"]}
    if int(snapshot["checkpoint_step"]) > int(old.get("checkpoint_step") or 0):
        return {
            "kind": "checkpoint",
            "stage": snapshot["stage"],
            "step": snapshot["checkpoint_step"],
            "path": snapshot["checkpoint_path"],
        }
    return None


def _fingerprint(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _append_report(status: dict[str, Any], event: dict[str, Any]) -> None:
    fingerprint = _fingerprint(event)
    if status.get("last_reported_event_fingerprint") == fingerprint:
        return
    summary = json.dumps(event, ensure_ascii=False, sort_keys=True)
    with REPORT_PATH.open("a", encoding="utf-8") as stream:
        stream.write(
            f"\n### {_now()} — 01_train supervisor event\n\n"
            f"- `{summary}`\n"
        )
    status["last_reported_event_fingerprint"] = fingerprint


def _launch_locked(status: dict[str, Any], *, recovery: bool) -> dict[str, Any]:
    runner = status.setdefault("runner", {})
    if _alive(runner.get("supervisor_pid")) or _alive(runner.get("pid")):
        return {"action": "already_running", "status": status}
    SUPERVISOR_DIR.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), "supervise"]
    with INTERNAL_LOG.open("ab") as stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    runner.update(
        {
            "supervisor_pid": process.pid,
            "alive": True,
            "exit_code": None,
            "started_at": _now(),
            "command": "PYTHONPATH=src .venv/bin/python -u 01_train/run_all.py --resume",
        }
    )
    status["state"] = "recovering" if recovery else "starting"
    status["error"] = None
    _save_status(status)
    return {"action": "launched", "supervisor_pid": process.pid, "status": status}


def launch() -> dict[str, Any]:
    with _lock():
        return _launch_locked(_load_status(), recovery=False)


def supervise() -> int:
    with _lock():
        status = _load_status()
        status["runner"]["supervisor_pid"] = os.getpid()
        status["runner"]["alive"] = True
        _save_status(status)
    command = [sys.executable, "-u", str(ROOT / "01_train/run_all.py"), "--resume"]
    with RUN_LOG.open("ab", buffering=0) as stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        with _lock():
            status = _load_status()
            status["runner"]["pid"] = process.pid
            status["runner"]["alive"] = True
            status["state"] = "running"
            _save_status(status)
        return_code = process.wait()
    with _lock():
        status = _load_status()
        status["runner"].update(
            {"pid": None, "supervisor_pid": None, "alive": False, "exit_code": return_code}
        )
        status["state"] = "exited"
        _save_status(status)
    return return_code


def check(*, recover: bool, update_report: bool) -> dict[str, Any]:
    with _lock():
        status = _load_status()
        previous = copy.deepcopy(status)
        new_log, cursor = _read_new_log(status)
        status["log_cursors"][str(RUN_LOG)] = cursor
        runner_alive = _alive(status.get("runner", {}).get("pid")) or _alive(
            status.get("runner", {}).get("supervisor_pid")
        )
        status["runner"]["alive"] = runner_alive
        snapshot_error = None
        try:
            snapshot = _snapshot()
        except Exception as exc:
            snapshot = {
                "stage": status.get("stage", "unknown"),
                **(status.get("progress") or {}),
                "complete": False,
                "milestones": [],
            }
            snapshot_error = {
                "key": "state_inconsistent",
                "retryable": False,
                "message": f"{type(exc).__name__}: {exc}",
                "observed_at": _now(),
            }
        previous_step = int((status.get("progress") or {}).get("step") or 0)
        status["stage"] = snapshot["stage"]
        status["progress"] = {
            key: snapshot.get(key)
            for key in (
                "step",
                "checkpoint_step",
                "checkpoint_path",
                "swanlab_run_id",
                "early_gate",
                "branches",
            )
        }
        previous_total = sum(
            int(item.get("step") or 0)
            for item in ((previous.get("progress") or {}).get("branches") or {}).values()
        )
        snapshot_total = sum(
            int(item.get("step") or 0)
            for item in (snapshot.get("branches") or {}).values()
        )
        if int(snapshot.get("step") or 0) > previous_step or snapshot_total > previous_total:
            status["recovery"] = {
                "error_key": None,
                "attempts": 0,
                "max_attempts": MAX_RECOVERIES,
            }

        error = snapshot_error
        if not runner_alive and not snapshot.get("complete") and error is None:
            exit_code = status.get("runner", {}).get("exit_code")
            if exit_code not in (None, 0) or new_log:
                error = _classify_error(new_log or _tail(RUN_LOG))
                error["message"] = f"01_train exited with code {exit_code}"
        if snapshot.get("complete"):
            status["state"] = "complete"
            status["error"] = None
        elif error is not None:
            status["state"] = "blocked" if not error["retryable"] else "failed_retryable"
            status["error"] = error
        elif runner_alive:
            status["state"] = "running"

        event = _event(previous, snapshot, error)
        observed = set(status.setdefault("observed_milestones", []))
        new_milestones = [
            milestone
            for milestone in snapshot.get("milestones", [])
            if milestone["id"] not in observed
        ]
        if new_milestones and error is None and not snapshot.get("complete"):
            event = {
                "kind": "milestones",
                "stage": snapshot["stage"],
                "milestones": new_milestones,
            }
            observed.update(item["id"] for item in new_milestones)
            status["observed_milestones"] = sorted(observed)
        elif snapshot.get("complete"):
            observed.update(item["id"] for item in snapshot.get("milestones", []))
            status["observed_milestones"] = sorted(observed)
        if event is not None:
            event["observed_at"] = _now()
            fingerprint = _fingerprint({k: v for k, v in event.items() if k != "observed_at"})
            if fingerprint != status.get("last_event_fingerprint"):
                status["last_event"] = event
                status["last_event_fingerprint"] = fingerprint
                if update_report:
                    _append_report(status, event)
            else:
                event = None

        action = "none"
        if error is not None and error["retryable"] and recover:
            recovery = status.setdefault("recovery", {})
            if recovery.get("error_key") != error["key"]:
                recovery.update({"error_key": error["key"], "attempts": 0})
            if int(recovery.get("attempts", 0)) < MAX_RECOVERIES:
                recovery["attempts"] = int(recovery.get("attempts", 0)) + 1
                result = _launch_locked(status, recovery=True)
                action = result["action"]
            else:
                status["state"] = "blocked"
                status["error"]["key"] = "retry_limit_exceeded"
        status["last_checked_at"] = _now()
        _save_status(status)
        return {"action": action, "event": event, "status": status}


def initialize() -> dict[str, Any]:
    with _lock():
        status = _default_status()
        snapshot = _snapshot()
        status["stage"] = snapshot["stage"]
        status["progress"] = {
            key: snapshot.get(key)
            for key in (
                "step",
                "checkpoint_step",
                "checkpoint_path",
                "swanlab_run_id",
                "early_gate",
                "branches",
            )
        }
        status["state"] = "complete" if snapshot["complete"] else "ready"
        status["last_checked_at"] = _now()
        _save_status(status)
        return status


def repair_running_status() -> dict[str, Any]:
    """Clear a stale supervisor-only error when the recorded runner is alive."""

    with _lock():
        status = _load_status()
        alive = _alive(status.get("runner", {}).get("pid")) or _alive(
            status.get("runner", {}).get("supervisor_pid")
        )
        if not alive:
            raise RuntimeError("Recorded supervisor and runner processes are not alive")
        snapshot = _snapshot()
        status["runner"]["alive"] = True
        status["state"] = "running"
        status["stage"] = snapshot["stage"]
        status["progress"] = {
            key: snapshot.get(key)
            for key in (
                "step",
                "checkpoint_step",
                "checkpoint_path",
                "swanlab_run_id",
                "early_gate",
                "branches",
            )
        }
        status["error"] = None
        status["last_event"] = None
        status["last_event_fingerprint"] = None
        status["last_checked_at"] = _now()
        _save_status(status)
        return status


def mark_stopped() -> dict[str, Any]:
    """Record an externally stopped pipeline after verifying no runner is alive."""

    with _lock():
        status = _load_status()
        if _alive(status.get("runner", {}).get("pid")) or _alive(
            status.get("runner", {}).get("supervisor_pid")
        ):
            raise RuntimeError("Cannot mark 01_train stopped while a runner is alive")
        snapshot = _snapshot()
        status["runner"].update(
            {"pid": None, "supervisor_pid": None, "alive": False, "exit_code": None}
        )
        status["state"] = "complete" if snapshot["complete"] else "stopped"
        status["stage"] = snapshot["stage"]
        status["progress"] = {
            key: snapshot.get(key)
            for key in (
                "step",
                "checkpoint_step",
                "checkpoint_path",
                "swanlab_run_id",
                "early_gate",
                "branches",
            )
        }
        status["error"] = None
        status.setdefault("log_cursors", {})[str(RUN_LOG)] = (
            RUN_LOG.stat().st_size if RUN_LOG.exists() else 0
        )
        event = {
            "kind": "complete" if snapshot["complete"] else "stopped",
            "stage": snapshot["stage"],
            "observed_at": _now(),
        }
        status["last_event"] = event
        status["last_event_fingerprint"] = _fingerprint(
            {key: value for key, value in event.items() if key != "observed_at"}
        )
        status["last_checked_at"] = _now()
        _save_status(status)
        return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("initialize")
    subparsers.add_parser("launch")
    subparsers.add_parser("supervise")
    subparsers.add_parser("repair-running-status")
    subparsers.add_parser("mark-stopped")
    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--recover", action="store_true")
    check_parser.add_argument("--update-report", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "initialize":
        result: Any = initialize()
    elif args.command == "launch":
        result = launch()
    elif args.command == "supervise":
        raise SystemExit(supervise())
    elif args.command == "repair-running-status":
        result = repair_running_status()
    elif args.command == "mark-stopped":
        result = mark_stopped()
    else:
        result = check(recover=args.recover, update_report=args.update_report)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
