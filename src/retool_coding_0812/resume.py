"""Checkpoint discovery and local-artifact reconciliation for safe resume."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
from typing import Any


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint must be a JSON object: {source}")
    step = int(value.get("step", 0))
    if step < 1 or not value.get("state_path"):
        raise ValueError(f"Checkpoint lacks a positive step or state_path: {source}")
    return value


def latest_checkpoint(
    checkpoint_dir: str | Path,
    *,
    pattern: str = "*.json",
    expected_run_name: str | None = None,
) -> Path | None:
    """Return the highest-step valid checkpoint matching the requested run."""

    candidates: list[tuple[int, Path]] = []
    for path in sorted(Path(checkpoint_dir).glob(pattern)):
        try:
            record = load_checkpoint(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if expected_run_name is not None and record.get("run_name") != expected_run_name:
            continue
        candidates.append((int(record["step"]), path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def read_metric_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid metrics JSON at {source}:{line_number}") from exc
            if not isinstance(row, dict) or int(row.get("step", 0)) < 1:
                raise ValueError(f"Invalid metrics row at {source}:{line_number}")
            rows.append(row)
    return rows


def validate_metric_prefix(rows: list[dict[str, Any]], checkpoint_step: int) -> list[dict[str, Any]]:
    """Require exactly one authoritative metrics row for every committed step."""

    prefix = [row for row in rows if int(row["step"]) <= checkpoint_step]
    actual = [int(row["step"]) for row in prefix]
    expected = list(range(1, checkpoint_step + 1))
    if actual != expected:
        raise ValueError(
            "Local metrics are not a contiguous, unique prefix through checkpoint "
            f"step {checkpoint_step}: got {actual[:5]}...{actual[-5:]}"
        )
    return prefix


def _recovery_dir(run_dir: Path, checkpoint_step: int) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return run_dir / "recovery" / f"{stamp}-rollback-to-step-{checkpoint_step}"


def reconcile_local_artifacts(
    *,
    run_dir: str | Path,
    checkpoint_step: int,
    checkpoint_dir: str | Path | None = None,
    checkpoint_pattern: str = "*.json",
    expected_run_name: str | None = None,
    trajectory_dir_name: str | None = "trajectories",
) -> list[dict[str, Any]]:
    """Roll local evidence back to the last remotely committed checkpoint.

    Metrics and trajectories newer than the checkpoint are not authoritative:
    the corresponding remote optimizer state was not saved.  They are moved to
    a timestamped recovery directory rather than deleted.
    """

    root = Path(run_dir)
    metrics_path = root / "metrics.jsonl"
    rows = read_metric_rows(metrics_path)
    prefix = validate_metric_prefix(rows, checkpoint_step)
    stale_rows = [row for row in rows if int(row["step"]) > checkpoint_step]

    stale_trajectories: list[Path] = []
    if trajectory_dir_name is not None:
        trajectory_root = root / trajectory_dir_name
        if trajectory_root.exists():
            for path in trajectory_root.glob("step-*"):
                try:
                    step = int(path.name.removeprefix("step-"))
                except ValueError:
                    continue
                if step > checkpoint_step:
                    stale_trajectories.append(path)

    stale_checkpoints: list[Path] = []
    if checkpoint_dir is not None:
        for path in Path(checkpoint_dir).glob(checkpoint_pattern):
            try:
                record = load_checkpoint(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if expected_run_name is not None and record.get("run_name") != expected_run_name:
                continue
            if int(record["step"]) > checkpoint_step:
                stale_checkpoints.append(path)

    if not (stale_rows or stale_trajectories or stale_checkpoints):
        return prefix

    recovery = _recovery_dir(root, checkpoint_step)
    recovery.mkdir(parents=True, exist_ok=False)
    if metrics_path.exists():
        shutil.copy2(metrics_path, recovery / "metrics.jsonl")
        with metrics_path.open("w", encoding="utf-8") as stream:
            for row in prefix:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    for path in stale_trajectories:
        destination = recovery / (trajectory_dir_name or "trajectories") / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), destination)

    for path in stale_checkpoints:
        destination = recovery / "checkpoints" / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), destination)

    return prefix
