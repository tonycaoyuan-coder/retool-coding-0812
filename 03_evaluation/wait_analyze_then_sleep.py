"""Keep the Mac awake, wait for evaluation, run GPT-5.6 Sol xhigh analysis."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVAL_STATUS = ROOT / "artifacts/evaluation-l16k-t24k/background-status.json"
OUTPUT_DIR = ROOT / "artifacts/analysis/l16k-t24k"
GUARDIAN_STATUS = OUTPUT_DIR / "sleep-guardian-status.json"
GUARDIAN_LOCK = OUTPUT_DIR / ".sleep-guardian.lock"
PROMPT_PATH = ROOT / "docs/investigations/length-ablation-analysis-prompt.md"
REPORT_PATH = ROOT / "docs/investigations/length-ablation-l16k-vs-l10k-report.md"
FINAL_MESSAGE_PATH = OUTPUT_DIR / "gpt56-sol-xhigh-final-message.txt"
CODEX_LOG_PATH = OUTPUT_DIR / "gpt56-sol-xhigh-codex.log"
CODEX_ERROR_PATH = OUTPUT_DIR / "gpt56-sol-xhigh-codex.err.log"
EVAL_ORCHESTRATOR = ROOT / "03_evaluation/run_length_ablation_background.py"
PYTHON = ROOT / ".venv/bin/python"
CODEX = Path("/Applications/ChatGPT.app/Contents/Resources/codex")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def write_status(state: str, **values: Any) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    previous = read_json(GUARDIAN_STATUS)
    payload = {
        **previous,
        "state": state,
        "pid": os.getpid(),
        "updated_at": now(),
        **values,
    }
    temporary = GUARDIAN_STATUS.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(GUARDIAN_STATUS)


def validate_inputs() -> None:
    missing = [
        str(path)
        for path in (EVAL_STATUS, PROMPT_PATH, EVAL_ORCHESTRATOR, PYTHON, CODEX)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing required paths: {missing}")


def wait_for_evaluation() -> None:
    resume_attempts = 0
    last_resume_at = 0.0
    while True:
        status = read_json(EVAL_STATUS)
        state = status.get("state")
        write_status(
            "waiting_for_evaluation",
            evaluation_state=state,
            evaluation_status_path=str(EVAL_STATUS),
            resume_attempts=resume_attempts,
        )
        if state == "complete":
            return
        if state == "failed" and time.monotonic() - last_resume_at >= 600:
            if resume_attempts >= 3:
                write_status(
                    "blocked_awake",
                    error="Evaluation failed after three automatic resume attempts; computer remains awake.",
                )
            else:
                resume_attempts += 1
                last_resume_at = time.monotonic()
                write_status("resuming_evaluation", resume_attempts=resume_attempts)
                with (OUTPUT_DIR / "evaluation-resume.log").open("a", encoding="utf-8") as out:
                    with (OUTPUT_DIR / "evaluation-resume.err.log").open(
                        "a", encoding="utf-8"
                    ) as err:
                        subprocess.run(
                            [str(PYTHON), str(EVAL_ORCHESTRATOR)],
                            cwd=ROOT,
                            stdout=out,
                            stderr=err,
                            check=False,
                        )
        time.sleep(30)


def run_codex_analysis() -> None:
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    command = [
        str(CODEX),
        "exec",
        "--model",
        "gpt-5.6-sol",
        "--config",
        'model_reasoning_effort="xhigh"',
        "--approve-for-me",
        "--cd",
        str(ROOT),
        "--skip-git-repo-check",
        "--output-last-message",
        str(FINAL_MESSAGE_PATH),
        "-",
    ]
    write_status(
        "analyzing",
        model="gpt-5.6-sol",
        reasoning_effort="xhigh",
        report_path=str(REPORT_PATH),
        analysis_started_at=now(),
    )
    with CODEX_LOG_PATH.open("a", encoding="utf-8") as out:
        with CODEX_ERROR_PATH.open("a", encoding="utf-8") as err:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                input=prompt,
                text=True,
                stdout=out,
                stderr=err,
                check=False,
            )
    if completed.returncode != 0:
        raise RuntimeError(f"Codex analysis exited with code {completed.returncode}")
    if not REPORT_PATH.exists() or REPORT_PATH.stat().st_size < 4_096:
        raise RuntimeError("Codex analysis did not produce a sufficiently complete report")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    validate_inputs()
    if args.validate_only:
        print(
            json.dumps(
                {
                    "ok": True,
                    "evaluation_state": read_json(EVAL_STATUS).get("state"),
                    "model": "gpt-5.6-sol",
                    "reasoning_effort": "xhigh",
                    "report_path": str(REPORT_PATH),
                },
                ensure_ascii=False,
            )
        )
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with GUARDIAN_LOCK.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        write_status("starting", started_at=now(), keep_awake=True)
        wait_for_evaluation()
        attempts = 0
        while True:
            attempts += 1
            try:
                run_codex_analysis()
                break
            except Exception as exc:  # Keep the machine awake if analysis cannot finish.
                write_status(
                    "analysis_retry_wait",
                    analysis_attempts=attempts,
                    error=f"{type(exc).__name__}: {exc}",
                )
                if attempts >= 3:
                    write_status(
                        "blocked_awake",
                        error="GPT-5.6 Sol xhigh analysis failed three times; computer remains awake.",
                    )
                    while True:
                        time.sleep(30)
                time.sleep(60)
        write_status(
            "analysis_complete_sleep_pending",
            keep_awake=False,
            analysis_completed_at=now(),
            report_path=str(REPORT_PATH),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
