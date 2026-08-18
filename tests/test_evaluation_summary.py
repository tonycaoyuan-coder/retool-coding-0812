from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _summary_module():
    source = ROOT / "03_evaluation/summarize.py"
    spec = importlib.util.spec_from_file_location("retool_0812_summary", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_run(root: Path, prompt: str) -> Path:
    run_dir = root / prompt
    (run_dir / "summary").mkdir(parents=True)
    cells = []
    deltas = []
    mcnemar = []
    for index, train in enumerate(("base", "c0", "c1", "c2")):
        cells.append(
            {
                "model_id": train,
                "benchmark_id": "benchmark",
                "train_prompt": train,
                "test_prompt": prompt.upper(),
                "checkpoint_step": 0 if train == "base" else 100,
                "value": 0.5 + index / 10,
                "tasks": 200,
            }
        )
        if train != "base":
            deltas.append(
                {
                    "train_prompt": train,
                    "test_prompt": prompt.upper(),
                    "delta": index / 10,
                    "ci_low": 0.0,
                    "ci_high": 0.2,
                    "wins": index,
                    "ties": 200 - index,
                    "losses": 0,
                }
            )
            mcnemar.append(
                {
                    "train_prompt": train,
                    "test_prompt": prompt.upper(),
                    "trained_only": index,
                    "base_only": 0,
                    "discordant": index,
                    "p_value": 1.0,
                }
            )
    summary = {
        "counts": {"completed": 800, "failed": 0, "pending": 0, "running": 0},
        "matrix_analysis": {
            "cells": cells,
            "deltas_vs_base": deltas,
            "mcnemar_vs_base": mcnemar,
            "paired": True,
        },
    }
    (run_dir / "summary/metrics.json").write_text(json.dumps(summary), encoding="utf-8")
    with sqlite3.connect(run_dir / "state.sqlite") as connection:
        connection.execute("CREATE TABLE tasks(model_id TEXT, sample_id TEXT, status TEXT)")
        connection.executemany(
            "INSERT INTO tasks VALUES (?, ?, 'completed')",
            [
                (model, f"task-{sample}")
                for model in ("base", "c0", "c1", "c2")
                for sample in range(200)
            ],
        )
    return run_dir


class EvaluationSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _summary_module()

    def test_three_prompt_shards_merge_to_complete_matrix(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            result = self.module.merge_summaries(
                [_make_run(root, prompt) for prompt in ("c0", "c1", "c2")]
            )
            matrix = result["matrix_analysis"]
            self.assertEqual(result["counts"]["completed"], 2400)
            self.assertEqual(len(matrix["cells"]), 12)
            self.assertEqual(len(matrix["deltas_vs_base"]), 9)
            self.assertEqual(len(matrix["mcnemar_vs_base"]), 9)
            self.assertEqual(len(matrix["robustness"]), 3)

    def test_duplicate_prompt_shard_is_rejected(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            c0 = _make_run(root, "c0")
            c1 = _make_run(root, "c1")
            with self.assertRaisesRegex(ValueError, "Duplicate final test-prompt shard"):
                self.module.merge_summaries([c0, c1, c0])


if __name__ == "__main__":
    unittest.main()
