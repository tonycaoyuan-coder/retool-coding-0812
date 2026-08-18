"""Small fail-fast process orchestrator for independent experiment branches."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Sequence


Command = Sequence[str]


def _stop_process_group(process: subprocess.Popen[bytes], timeout: float = 10.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait()


def run_parallel(
    commands: Sequence[Command],
    *,
    cwd: Path,
    labels: Sequence[str] | None = None,
) -> None:
    """Run independent commands concurrently and terminate siblings on failure."""

    if labels is not None and len(labels) != len(commands):
        raise ValueError("labels and commands must have the same length")
    names = list(labels or (f"job-{index}" for index in range(len(commands))))
    processes: list[tuple[str, list[str], subprocess.Popen[bytes]]] = []

    def handle_termination(signum: int, _frame: object) -> None:
        raise SystemExit(128 + signum)

    previous_sigterm = signal.signal(signal.SIGTERM, handle_termination)
    try:
        for name, raw_command in zip(names, commands, strict=True):
            command = list(raw_command)
            if not command:
                raise ValueError(f"Parallel command {name!r} is empty")
            print(f"[parallel] starting {name}: {subprocess.list2cmdline(command)}", flush=True)
            processes.append(
                (
                    name,
                    command,
                    subprocess.Popen(command, cwd=cwd, start_new_session=True),
                )
            )

        pending = {process.pid for _, _, process in processes}
        while pending:
            for name, command, process in processes:
                if process.pid not in pending:
                    continue
                return_code = process.poll()
                if return_code is None:
                    continue
                pending.remove(process.pid)
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, command)
                print(f"[parallel] completed {name}", flush=True)
            if pending:
                time.sleep(0.1)
    except BaseException:
        for _, _, process in processes:
            _stop_process_group(process)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
