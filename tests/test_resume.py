from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from retool_coding_0812.resume import (
    latest_checkpoint,
    read_metric_rows,
    reconcile_local_artifacts,
)
from retool_coding_0812.eval_resume import evaluation_run_lock
from retool_coding_0812.settings import branch_args


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


class ResumeTests(unittest.TestCase):
    def test_evaluation_run_lock_rejects_duplicate_shard(self) -> None:
        with evaluation_run_lock(
            Path("configs/generated/checkpoint_dev/c0.yaml")
        ):
            with self.assertRaisesRegex(RuntimeError, "already running"):
                with evaluation_run_lock(
                    Path("configs/generated/checkpoint_dev/c0.yaml")
                ):
                    pass

    def test_branch_resume_does_not_also_load_shared_sft(self) -> None:
        checkpoint = Path("checkpoint.json")
        args = branch_args("c1", resume_checkpoint=checkpoint, recover=True)
        self.assertEqual(args.resume_checkpoint, checkpoint)
        self.assertIsNone(args.initial_state_manifest)
        self.assertTrue(args.recover)

    def test_latest_checkpoint_uses_highest_step_for_run(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            for run_name, step in (("run-a", 20), ("run-a", 60), ("run-b", 100)):
                _write_json(
                    root / f"{run_name}-step-{step}.json",
                    {"run_name": run_name, "step": step, "state_path": f"trio://{step}"},
                )
            (root / "run-a-step-80.json").write_text("{", encoding="utf-8")
            actual = latest_checkpoint(
                root, pattern="run-a-step-*.json", expected_run_name="run-a"
            )
            self.assertEqual(actual, root / "run-a-step-60.json")

    def test_reconcile_rolls_back_uncommitted_local_artifacts(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            run_dir = root / "run"
            checkpoint_dir = root / "checkpoints"
            run_name = "run-a"
            metrics_path = run_dir / "metrics.jsonl"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            for step in range(1, 24):
                with metrics_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"step": step, "reward/mean": step / 100}) + "\n")
            for step in (20, 22):
                _write_json(
                    checkpoint_dir / f"{run_name}-step-{step}.json",
                    {"run_name": run_name, "step": step, "state_path": f"trio://{step}"},
                )
            for step in (20, 21, 22, 23):
                trajectory = run_dir / "trajectories" / f"step-{step:04d}" / "item.json.gz"
                trajectory.parent.mkdir(parents=True, exist_ok=True)
                trajectory.write_bytes(b"trajectory")

            rows = reconcile_local_artifacts(
                run_dir=run_dir,
                checkpoint_step=20,
                checkpoint_dir=checkpoint_dir,
                checkpoint_pattern=f"{run_name}-step-*.json",
                expected_run_name=run_name,
            )

            self.assertEqual([row["step"] for row in rows], list(range(1, 21)))
            self.assertEqual(
                [row["step"] for row in read_metric_rows(run_dir / "metrics.jsonl")],
                list(range(1, 21)),
            )
            self.assertTrue((run_dir / "trajectories/step-0020/item.json.gz").exists())
            self.assertFalse((run_dir / "trajectories/step-0021").exists())
            self.assertFalse((checkpoint_dir / f"{run_name}-step-22.json").exists())
            recoveries = list((run_dir / "recovery").iterdir())
            self.assertEqual(len(recoveries), 1)
            self.assertTrue((recoveries[0] / "metrics.jsonl").exists())
            self.assertTrue((recoveries[0] / "trajectories/step-0023/item.json.gz").exists())
            self.assertTrue((recoveries[0] / "checkpoints" / f"{run_name}-step-22.json").exists())

    def test_reconcile_rejects_gaps_in_committed_metrics(self) -> None:
        with TemporaryDirectory() as raw:
            run_dir = Path(raw)
            metrics = run_dir / "metrics.jsonl"
            metrics.write_text('{"step": 1}\n{"step": 3}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contiguous"):
                reconcile_local_artifacts(run_dir=run_dir, checkpoint_step=3)


if __name__ == "__main__":
    unittest.main()
