"""JSON-in/JSON-out worker executed only inside the sandbox image."""

from __future__ import annotations

import ast
import contextlib
import io
import json
import signal
import sys
import time
import traceback
from decimal import Decimal, InvalidOperation
from typing import Any


MAX_TEXT = 4096
MAX_CAPTURE = 64 * 1024
IMPORT_PRELUDE = """from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from typing import *
import string, re, datetime, collections, heapq, bisect, copy, math, random
import statistics, itertools, functools, operator, io, sys, json
sys.setrecursionlimit(50000)
"""


class CaseTimeout(Exception):
    pass


class LimitedStringIO(io.StringIO):
    """Bound captured output so hostile programs cannot grow an in-memory buffer forever."""

    def __init__(self, limit: int = MAX_CAPTURE) -> None:
        super().__init__()
        self.limit = limit
        self.truncated = False

    def write(self, value: str) -> int:
        requested = len(value)
        remaining = self.limit - self.tell()
        if remaining > 0:
            super().write(value[:remaining])
        if requested > max(remaining, 0):
            self.truncated = True
        return requested

    def captured(self) -> str:
        value = self.getvalue()
        return value + ("\n[output truncated]\n" if self.truncated else "")


def _timeout_handler(signum, frame):
    del signum, frame
    raise CaseTimeout("execution timed out")


def capped(value: object, limit: int = MAX_TEXT) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit // 2] + "\n[... truncated ...]\n" + text[-limit // 2 :]


@contextlib.contextmanager
def timer(seconds: float):
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, max(float(seconds), 0.01))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def execute_tool(payload: dict[str, Any]) -> dict[str, Any]:
    stdout = LimitedStringIO()
    stderr = LimitedStringIO()
    previous_stdin = sys.stdin
    started = time.perf_counter()
    status = "success"
    try:
        sys.stdin = io.StringIO(str(payload.get("stdin", "")))
        with timer(float(payload.get("timeout_seconds", 5.0))):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                namespace = {"__name__": "__main__", "__builtins__": __builtins__}
                exec(compile(str(payload.get("code", "")), "<run_python>", "exec"), namespace)
    except CaseTimeout:
        status = "time_limit"
    except BaseException:
        status = "runtime_error"
        traceback.print_exc(file=stderr)
    finally:
        sys.stdin = previous_stdin
    return {
        "status": status,
        "stdout": capped(stdout.captured()),
        "stderr": capped(stderr.captured()),
        "execution_seconds": time.perf_counter() - started,
    }


def stripped_lines(value: str) -> list[str]:
    return [line.strip() for line in value.strip().splitlines()]


def decimal_tokens(value: str) -> list[Decimal] | None:
    try:
        return [Decimal(token) for token in value.split()]
    except InvalidOperation:
        return None


def stdio_equal(actual: str, expected: str) -> bool:
    left = stripped_lines(actual)
    right = stripped_lines(expected)
    if len(left) != len(right):
        return False
    for actual_line, expected_line in zip(left, right):
        if actual_line == expected_line:
            continue
        actual_numbers = decimal_tokens(actual_line)
        expected_numbers = decimal_tokens(expected_line)
        if actual_numbers is None or expected_numbers is None or actual_numbers != expected_numbers:
            return False
    return True


def functional_args(value: str) -> list[Any]:
    return [json.loads(line) for line in value.splitlines()]


def compile_candidate(code: str) -> Any:
    return compile(IMPORT_PRELUDE + "\n" + code, "<candidate>", "exec")


def run_stdio(compiled: Any, input_text: str, timeout_seconds: float) -> str:
    stdout = LimitedStringIO()
    previous_stdin = sys.stdin
    try:
        sys.stdin = io.StringIO(input_text)
        namespace = {"__name__": "__main__", "__builtins__": __builtins__}
        with timer(timeout_seconds), contextlib.redirect_stdout(stdout):
            try:
                exec(compiled, namespace)
            except SystemExit:
                pass
    finally:
        sys.stdin = previous_stdin
    return stdout.captured()


def prepare_functional(
    compiled: Any,
    fn_name: str,
    timeout_seconds: float,
) -> Any:
    namespace = {"__name__": "candidate", "__builtins__": __builtins__}
    stdout = LimitedStringIO()
    stderr = LimitedStringIO()
    with timer(timeout_seconds), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        exec(compiled, namespace)
        target: Any = namespace.get("Solution", namespace)
        if isinstance(target, type):
            target = target()
        function = getattr(target, fn_name) if not isinstance(target, dict) else target.get(fn_name)
        if function is None:
            raise AttributeError(f"missing function {fn_name}")
    return function


def run_functional(function: Any, input_text: str, timeout_seconds: float) -> Any:
    stdout = LimitedStringIO()
    stderr = LimitedStringIO()
    with timer(timeout_seconds), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        result = function(*functional_args(input_text))
    return list(result) if isinstance(result, tuple) else result


def judge(payload: dict[str, Any]) -> dict[str, Any]:
    tests = list(payload.get("tests") or [])
    public_total = sum(item.get("source") == "public" for item in tests)
    private_total = sum(item.get("source") == "private" for item in tests)
    result = {
        "passed": 0,
        "total": len(tests),
        "public_passed": 0,
        "public_total": public_total,
        "private_passed": 0,
        "private_total": private_total,
        "status": "wrong_answer",
        "first_failure": "",
    }
    started = time.perf_counter()
    try:
        compiled = compile_candidate(str(payload.get("code", "")))
    except BaseException as exc:
        result.update(status="compile_error", first_failure=capped(repr(exc)))
        result["execution_seconds"] = time.perf_counter() - started
        return result
    fn_name = payload.get("fn_name")
    timeout_seconds = float(payload.get("case_timeout_seconds", 3.0))
    functional = None
    if fn_name:
        try:
            functional = prepare_functional(compiled, str(fn_name), timeout_seconds)
        except CaseTimeout:
            result.update(status="time_limit", first_failure="candidate setup time limit")
            result["execution_seconds"] = time.perf_counter() - started
            return result
        except BaseException as exc:
            result.update(status="runtime_error", first_failure=capped(repr(exc)))
            result["execution_seconds"] = time.perf_counter() - started
            return result
    deadline = started + float(payload.get("judge_timeout_seconds", 29.0))
    first_failure_status = ""
    for index, case in enumerate(tests):
        source = str(case.get("source", "private"))
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            if not first_failure_status:
                first_failure_status = "time_limit"
                result["first_failure"] = f"case {index}: whole-judge time limit"
            break
        try:
            if fn_name:
                actual = run_functional(
                    functional,
                    str(case.get("input", "")),
                    min(timeout_seconds, remaining),
                )
                expected = json.loads(str(case.get("output", "")))
                passed = actual == expected
            else:
                actual = run_stdio(
                    compiled,
                    str(case.get("input", "")),
                    min(timeout_seconds, remaining),
                )
                expected = str(case.get("output", ""))
                passed = stdio_equal(actual, expected)
            if not passed:
                if not first_failure_status:
                    first_failure_status = "wrong_answer"
                    result["first_failure"] = capped(
                        f"case {index}: expected={expected!r}, actual={actual!r}"
                    )
                continue
        except CaseTimeout:
            if not first_failure_status:
                first_failure_status = "time_limit"
                result["first_failure"] = f"case {index}: time limit"
            # Re-running the same non-terminating program on every hidden case would
            # consume the whole judge budget. Remaining cases count as failed.
            break
        except BaseException as exc:
            if not first_failure_status:
                first_failure_status = "runtime_error"
                result["first_failure"] = capped(f"case {index}: {exc!r}")
            continue
        result["passed"] += 1
        result[f"{source}_passed"] += 1
    if result["passed"] == result["total"]:
        result["status"] = "pass"
    else:
        result["status"] = first_failure_status or "wrong_answer"
    result["execution_seconds"] = time.perf_counter() - started
    return result


def main() -> None:
    try:
        payload = json.load(sys.stdin)
        mode = payload.get("mode")
        if mode == "tool":
            result = execute_tool(payload)
        elif mode == "judge":
            result = judge(payload)
        else:
            raise ValueError(f"unknown worker mode {mode!r}")
    except BaseException as exc:
        result = {"worker_error": capped(repr(exc)), "traceback": capped(traceback.format_exc())}
    sys.stdout.write(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
