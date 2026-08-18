"""Frozen LiveCodeBench-v6 normalization and deterministic temporal splits."""

from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import pickle
import zlib
from dataclasses import asdict, dataclass, field
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping


DATASET_ID = "livecodebench/code_generation_lite"
DATASET_CONFIG = "v6"
DATASET_REVISION = "a16d03780493b939b3601fb9da2ac3ed2b23caa2"


@dataclass(frozen=True)
class TestCase:
    input: str
    output: str
    testtype: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TestCase":
        testtype = str(value.get("testtype", "")).lower()
        if testtype not in {"stdin", "functional"}:
            raise ValueError(f"Unsupported test type: {testtype!r}")
        return cls(
            input=str(value.get("input", "")),
            output=str(value.get("output", "")),
            testtype=testtype,
        )


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if value is None or value == "":
        return {}
    decoded = json.loads(str(value))
    if not isinstance(decoded, dict):
        raise ValueError("Expected a JSON object")
    return decoded


def _test_rows(value: Any, *, allow_legacy_pickle: bool = False) -> list[dict[str, Any]]:
    if isinstance(value, list):
        rows = value
    else:
        raw = str(value or "")
        try:
            rows = json.loads(raw)
        except json.JSONDecodeError:
            if not allow_legacy_pickle:
                raise
            rows = pickle.loads(zlib.decompress(base64.b64decode(raw.encode("utf-8"))))
        if isinstance(rows, (str, bytes, bytearray)):
            rows = json.loads(rows)
    if not isinstance(rows, list):
        raise ValueError("Test cases must decode to a list")
    return [dict(item) for item in rows]


@dataclass(frozen=True)
class LCBExample:
    instance_id: str
    question_title: str
    question_content: str
    starter_code: str
    platform: str
    contest_id: str
    contest_date: str
    difficulty: str
    fn_name: str | None
    public_tests: tuple[TestCase, ...]
    private_tests: tuple[TestCase, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def test_type(self) -> str:
        return "functional" if self.fn_name else "stdin"

    @property
    def all_tests(self) -> tuple[TestCase, ...]:
        return (*self.public_tests, *self.private_tests)

    @classmethod
    def from_mapping(cls, item: Mapping[str, Any]) -> "LCBExample":
        metadata = _json_object(item.get("metadata"))
        fn_name = item.get("fn_name", metadata.get("func_name"))
        fn_name = str(fn_name).strip() if fn_name else None
        public_value = item.get("public_tests", item.get("public_test_cases", []))
        private_value = item.get("private_tests", item.get("private_test_cases", []))
        public_tests = tuple(
            TestCase.from_mapping(value) for value in _test_rows(public_value)
        )
        private_tests = tuple(
            TestCase.from_mapping(value)
            for value in _test_rows(private_value, allow_legacy_pickle=True)
        )
        question_id = str(item.get("instance_id", item.get("question_id", ""))).strip()
        contest_id = str(item.get("contest_id", "")).strip()
        platform = str(item.get("platform", "")).strip().lower()
        instance_id = question_id or f"{platform}-{contest_id}"
        required = {
            "instance_id": instance_id,
            "question_content": str(item.get("question_content", "")).strip(),
            "contest_date": str(item.get("contest_date", "")).strip(),
            "tests": (*public_tests, *private_tests),
        }
        missing = [key for key, value in required.items() if not value]
        if missing:
            raise ValueError(f"Task {instance_id or '<unknown>'} missing: {', '.join(missing)}")
        expected_type = "functional" if fn_name else "stdin"
        wrong = [case.testtype for case in (*public_tests, *private_tests) if case.testtype != expected_type]
        if wrong:
            raise ValueError(f"Task {instance_id} mixes incompatible test types")
        return cls(
            instance_id=instance_id,
            question_title=str(item.get("question_title", "")).strip(),
            question_content=str(item.get("question_content", "")).strip(),
            starter_code=str(item.get("starter_code", "") or ""),
            platform=platform,
            contest_id=contest_id,
            contest_date=str(item.get("contest_date", "")).strip(),
            difficulty=str(item.get("difficulty", "")).strip().lower(),
            fn_name=fn_name,
            public_tests=public_tests,
            private_tests=private_tests,
            metadata=metadata,
        )

    def to_json(self) -> dict[str, Any]:
        value = asdict(self)
        value["public_tests"] = [asdict(item) for item in self.public_tests]
        value["private_tests"] = [asdict(item) for item in self.private_tests]
        value["metadata"] = dict(self.metadata)
        return value

    def prompt_record(self) -> dict[str, Any]:
        """Return metadata safe to persist in model trajectories."""
        return {
            "instance_id": self.instance_id,
            "question_title": self.question_title,
            "question_content": self.question_content,
            "starter_code": self.starter_code,
            "platform": self.platform,
            "contest_id": self.contest_id,
            "contest_date": self.contest_date,
            "difficulty": self.difficulty,
            "test_type": self.test_type,
        }


def stable_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _open_text(path: Path, mode: str):
    if path.suffix == ".gz":
        if mode == "w":
            with path.open("wb") as raw:
                with gzip.GzipFile(
                    filename="",
                    mode="wb",
                    fileobj=raw,
                    mtime=0,
                ) as compressed:
                    with io.TextIOWrapper(compressed, encoding="utf-8") as stream:
                        yield stream
            return
        with gzip.open(path, mode + "t", encoding="utf-8") as stream:
            yield stream
        return
    with path.open(mode, encoding="utf-8") as stream:
        yield stream


def load_examples(path: str | Path) -> list[LCBExample]:
    source = Path(path)
    examples: list[LCBExample] = []
    with _open_text(source, "r") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                examples.append(LCBExample.from_mapping(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"Invalid row {line_number} in {source}") from exc
    if not examples:
        raise ValueError(f"No examples in {source}")
    return examples


def take_training_batch(
    examples: list[LCBExample], *, step: int, questions_per_batch: int
) -> list[LCBExample]:
    start = step * questions_per_batch
    end = start + questions_per_batch
    if end > len(examples):
        raise ValueError("Training schedule would repeat or overrun the frozen task list")
    return examples[start:end]
