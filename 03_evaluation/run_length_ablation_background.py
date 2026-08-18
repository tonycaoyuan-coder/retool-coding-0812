"""Resume L16K/T24K evaluation and finalize all analysis artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from myeval.config import load_config

from retool_coding_0812.settings import ROOT


TAG = "l16k-t24k"
CONFIG_DIR = ROOT / f"configs/generated/evaluation-{TAG}"
BASELINE_CONFIG_DIR = ROOT / "configs/generated/evaluation"
OUTPUT_ROOT = ROOT / f"artifacts/evaluation-{TAG}"
ANALYSIS_DIR = ROOT / "artifacts/analysis" / TAG
STATUS_PATH = OUTPUT_ROOT / "background-status.json"
PROMPTS = ("c0", "c1", "c2")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(state: str, **values: Any) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    previous: dict[str, Any] = {}
    if STATUS_PATH.exists():
        try:
            previous = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = {}
    if state == "running":
        for key in (
            "failed_at",
            "error",
            "traceback",
            "completed_at",
            "new_runs",
            "baseline_runs",
            "analysis_dir",
        ):
            previous.pop(key, None)
    elif state == "complete":
        for key in ("failed_at", "error", "traceback"):
            previous.pop(key, None)
    payload = {
        **previous,
        "state": state,
        "pid": os.getpid(),
        "updated_at": now(),
        **values,
    }
    temporary = STATUS_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(STATUS_PATH)


def run(command: list[str]) -> None:
    print("[background]", " ".join(command), flush=True)
    environment = os.environ.copy()
    python_path = str(ROOT / "src")
    if environment.get("PYTHONPATH"):
        python_path = f"{python_path}{os.pathsep}{environment['PYTHONPATH']}"
    environment["PYTHONPATH"] = python_path
    subprocess.run(command, cwd=ROOT, check=True, env=environment)


def matching_complete_run(config_path: Path) -> Path:
    config = load_config(config_path)
    candidates: list[Path] = []
    for manifest_path in config.execution.output_dir.glob("*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (
            manifest.get("config_fingerprint") == config.config_fingerprint
            and manifest.get("status") == "complete"
        ):
            candidates.append(manifest_path.parent)
    if not candidates:
        raise RuntimeError(f"No complete run matches {config_path}")
    return max(candidates, key=lambda path: path.name)


def main() -> None:
    python = sys.executable
    write_status(
        "running",
        started_at=now(),
        config_dir=str(CONFIG_DIR.resolve()),
        log_path=str((OUTPUT_ROOT / "background.log").resolve()),
    )
    try:
        run(
            [
                python,
                str(ROOT / "03_evaluation/run.py"),
                "--config-dir",
                str(CONFIG_DIR),
                "--resume",
            ]
        )
        new_runs = [matching_complete_run(CONFIG_DIR / f"{prompt}.yaml") for prompt in PROMPTS]
        baseline_runs = [
            matching_complete_run(BASELINE_CONFIG_DIR / f"{prompt}.yaml") for prompt in PROMPTS
        ]
        ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        run(
            [
                python,
                str(ROOT / "03_evaluation/summarize.py"),
                "--myeval-runs",
                *(str(path) for path in new_runs),
                "--output",
                str(ANALYSIS_DIR / "final-matrix.md"),
            ]
        )
        run(
            [
                python,
                str(ROOT / "03_evaluation/audit_length_ablation.py"),
                "--myeval-runs",
                *(str(path) for path in new_runs),
                "--config-dir",
                str(CONFIG_DIR),
                "--baseline-config-dir",
                str(BASELINE_CONFIG_DIR),
                "--max-assistant-tokens",
                "16384",
                "--max-trajectory-tokens",
                "24576",
                "--timeout-seconds",
                "600",
                "--output",
                str(ANALYSIS_DIR / "audit.json"),
            ]
        )
        run(
            [
                python,
                str(ROOT / "03_evaluation/analyze_length_ablation.py"),
                "--new-runs",
                *(str(path) for path in new_runs),
                "--baseline-runs",
                *(str(path) for path in baseline_runs),
                "--output-dir",
                str(ANALYSIS_DIR),
            ]
        )
        write_status(
            "complete",
            completed_at=now(),
            new_runs=[str(path.resolve()) for path in new_runs],
            baseline_runs=[str(path.resolve()) for path in baseline_runs],
            analysis_dir=str(ANALYSIS_DIR.resolve()),
        )
        print(f"[background] complete: {ANALYSIS_DIR}", flush=True)
    except BaseException as exc:
        write_status(
            "failed",
            failed_at=now(),
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":
    main()
