from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _supervisor():
    source = ROOT / "01_train/supervisor.py"
    spec = importlib.util.spec_from_file_location("retool_0812_supervisor", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SupervisorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _supervisor()

    def test_error_classification_is_conservative(self) -> None:
        retryable = self.module._classify_error("HTTP 503 Service Unavailable")
        self.assertTrue(retryable["retryable"])
        self.assertEqual(retryable["key"], "http_transient")

        billing = self.module._classify_error("billing_insufficient_balance")
        self.assertFalse(billing["retryable"])
        self.assertEqual(billing["key"], "billing_insufficient_balance")

        unknown = self.module._classify_error("unexpected worker failure")
        self.assertFalse(unknown["retryable"])

    def test_event_detects_stage_gate_checkpoint_and_completion(self) -> None:
        previous = self.module._default_status()
        stage = {
            "stage": "c0",
            "step": 0,
            "checkpoint_step": 0,
            "checkpoint_path": None,
            "early_gate": None,
            "complete": False,
        }
        self.assertEqual(self.module._event(previous, stage, None)["kind"], "stage_changed")

        previous["stage"] = "c0"
        gate = {**stage, "step": 20, "early_gate": {"passed": True}}
        self.assertEqual(self.module._event(previous, gate, None)["kind"], "early_gate")

        previous["progress"]["early_gate"] = {"passed": True}
        checkpoint = {**gate, "checkpoint_step": 20, "checkpoint_path": "step-20.json"}
        self.assertEqual(
            self.module._event(previous, checkpoint, None)["kind"], "checkpoint"
        )

        complete = {**checkpoint, "stage": "complete", "complete": True}
        self.assertEqual(self.module._event(previous, complete, None)["kind"], "complete")

    def test_default_recovery_budget_is_three(self) -> None:
        status = self.module._default_status()
        self.assertEqual(status["recovery"]["max_attempts"], 3)
        self.assertEqual(status["recovery"]["attempts"], 0)

    def test_default_status_tracks_parallel_branches(self) -> None:
        status = self.module._default_status()
        self.assertEqual(status["progress"]["branches"], {})

    def test_eperm_process_probe_counts_as_alive(self) -> None:
        with patch.object(
            self.module.os,
            "kill",
            side_effect=PermissionError(self.module.errno.EPERM, "not permitted"),
        ):
            self.assertTrue(self.module._alive(12345))


if __name__ == "__main__":
    unittest.main()
