from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def _selection_module():
    source = ROOT / "02_checkpoint_dev/select_checkpoints.py"
    spec = importlib.util.spec_from_file_location("checkpoint_selection", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CheckpointSelectionTests(unittest.TestCase):
    def test_exact_average_tie_uses_worst_before_float_noise(self) -> None:
        module = _selection_module()
        candidates = [
            {
                "model_path": "trio://step-40",
                "train_variant": "c0",
                "seed": 42,
                "step": 40,
                "average": 0.5666666666666667,
                "pass_at_1": 0.5666666666666667,
                "worst": 0.56,
                "case_pass_rate": 0.64,
                "pass_count": 170,
                "pass_total": 300,
                "worst_pass_count": 56,
                "worst_pass_total": 100,
                "prompt_values": {"c0": 0.57, "c1": 0.57, "c2": 0.56},
            },
            {
                "model_path": "trio://step-80",
                "train_variant": "c0",
                "seed": 42,
                "step": 80,
                "average": 0.5666666666666668,
                "pass_at_1": 0.5666666666666668,
                "worst": 0.55,
                "case_pass_rate": 0.66,
                "pass_count": 170,
                "pass_total": 300,
                "worst_pass_count": 55,
                "worst_pass_total": 100,
                "prompt_values": {"c0": 0.60, "c1": 0.55, "c2": 0.55},
            },
        ]
        args = SimpleNamespace(
            myeval_runs=[Path("c0"), Path("c1"), Path("c2")],
            output=Path(self._testMethodName + ".json"),
            expected_tasks=100,
            required_steps=[40, 80],
        )
        branches = {
            "c0": candidates,
            "c1": [{**item, "train_variant": "c1"} for item in candidates],
            "c2": [{**item, "train_variant": "c2"} for item in candidates],
        }
        try:
            with patch.object(module, "candidates", side_effect=lambda path, **_: branches[path.name]):
                module.main(args)
            import json

            selected = json.loads(args.output.read_text(encoding="utf-8"))
            self.assertEqual(selected["c0"]["step"], 40)
        finally:
            args.output.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
