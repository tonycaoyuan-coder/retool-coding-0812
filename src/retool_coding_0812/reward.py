"""Execution outcome structures and the case-fraction reward."""

from __future__ import annotations

from dataclasses import dataclass, field


INVALID_FORMAT_REWARD = -0.1


@dataclass(frozen=True)
class JudgeResult:
    passed: int
    total: int
    public_passed: int
    public_total: int
    private_passed: int
    private_total: int
    status: str
    first_failure: str = ""
    execution_seconds: float = 0.0
    infrastructure_error: str | None = None
    details: dict[str, object] = field(default_factory=dict)

    @property
    def resolved(self) -> bool:
        return self.total > 0 and self.passed == self.total and not self.infrastructure_error

    @property
    def reward(self) -> float:
        if self.status == "invalid_format":
            return INVALID_FORMAT_REWARD
        return self.passed / max(self.total, 1)

    @property
    def public_rate(self) -> float:
        return self.public_passed / max(self.public_total, 1)

    @property
    def private_rate(self) -> float:
        return self.private_passed / max(self.private_total, 1)


def invalid_format_result(total: int = 0, public_total: int = 0, private_total: int = 0) -> JudgeResult:
    return JudgeResult(
        passed=0,
        total=total,
        public_passed=0,
        public_total=public_total,
        private_passed=0,
        private_total=private_total,
        status="invalid_format",
        first_failure="Final response is not exactly one non-empty Python code block.",
    )
