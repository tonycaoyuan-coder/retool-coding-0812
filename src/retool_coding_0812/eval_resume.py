"""Choose safe per-shard MyEval behavior for parallel resume commands."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
from typing import Iterator, Literal

from myeval.config import load_config
from myeval.runner import find_resumable_run, prepare_experiment


ResumeMode = Literal["run", "resume", "skip"]


@contextmanager
def evaluation_run_lock(config_path: Path) -> Iterator[None]:
    """Prevent duplicate processes from starting the same evaluation shard."""

    config = load_config(config_path)
    lock_path = config.execution.output_dir / ".run.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            stream.seek(0)
            owner = stream.read().strip() or "unknown"
            raise RuntimeError(
                f"Evaluation shard is already running (pid={owner}, lock={lock_path})"
            ) from exc
        stream.seek(0)
        stream.truncate()
        stream.write(str(os.getpid()))
        stream.flush()
        try:
            yield
        finally:
            stream.seek(0)
            stream.truncate()
            stream.flush()
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def resume_mode(config_path: Path, requested: bool) -> ResumeMode:
    if not requested:
        return "run"
    config = load_config(config_path)
    prepared = prepare_experiment(config)
    if find_resumable_run(config, prepared.code_fingerprint) is not None:
        return "resume"
    for manifest_path in config.execution.output_dir.glob("*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (
            manifest.get("config_fingerprint") == config.config_fingerprint
            and manifest.get("code_fingerprint") == prepared.code_fingerprint
            and manifest.get("status") == "complete"
        ):
            return "skip"
    return "run"
