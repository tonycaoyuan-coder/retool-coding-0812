from __future__ import annotations

import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from myeval.config import BenchmarkConfig

from retool_coding_0812.eval_config import checkpoint_dev_config, final_config
from retool_coding_0812.myeval_plugin import LCBCodegenMini
from retool_coding_0812.settings import branch_args, load_config


class FrozenConfigTests(unittest.TestCase):
    def test_lcb_snapshot_keeps_hidden_tests_out_of_persisted_sample(self) -> None:
        config = load_config()
        source = Path(config["inputs"]["formal_data_dir"]) / "checkpoint_dev.jsonl.gz"
        plugin = LCBCodegenMini(
            BenchmarkConfig(
                id="lcb_codegen_retool_0812",
                local_path=source,
                shots=[0],
                limit=1,
            )
        )
        snapshot = plugin.load()
        sample = snapshot.samples[0]

        self.assertEqual(sample.reference, {"instance_id": sample.id})
        self.assertTrue(plugin.example(sample.id).private_tests)
        self.assertNotIn("private_tests", json.dumps(sample.to_dict()))

    def test_seed_and_formal_scale_are_frozen(self) -> None:
        config = load_config()
        self.assertEqual(config["experiment"]["seed"], 42)
        self.assertEqual(config["grpo"]["steps"], 100)
        self.assertEqual(config["grpo"]["questions_per_step"], 4)
        self.assertEqual(config["grpo"]["group_size"], 8)
        self.assertEqual(config["checkpoint_dev"]["checkpoints"], [20, 40, 60, 80, 100])

    def test_branch_namespace_has_no_seed_override(self) -> None:
        args = branch_args("c2", gate_only=True)
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.max_assistant_tokens, 10240)
        self.assertEqual(args.max_trajectory_tokens, 20480)
        self.assertEqual(args.swanlab_mode, "online")

    def test_checkpoint_and_final_configs_are_complete(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            checkpoints = {
                f"{variant}-step-{step}": {
                    "model_path": f"trio://{variant}/{step}",
                    "train_variant": variant,
                    "step": step,
                    "seed": 42,
                }
                for variant in ("c0", "c1", "c2")
                for step in (20, 40, 60, 80, 100)
            }
            checkpoint_path = root / "checkpoints.json"
            checkpoint_path.write_text(json.dumps(checkpoints), encoding="utf-8")
            dev = checkpoint_dev_config("c1", checkpoint_path)
            self.assertEqual(len(dev["models"]), 5)
            self.assertEqual(len(dev["system_prompts"]["profiles"]), 3)
            self.assertTrue(dev["tracking"]["swanlab"])
            self.assertEqual(dev["tracking"]["mode"], "online")

            selected = {"base": None}
            for variant in ("c0", "c1", "c2"):
                selected[variant] = checkpoints[f"{variant}-step-100"]
            selected_path = root / "selected.json"
            selected_path.write_text(json.dumps(selected), encoding="utf-8")
            final = final_config(selected_path)
            self.assertEqual(len(final["models"]), 4)
            self.assertEqual(final["benchmarks"][0]["limit"], 200)
            self.assertEqual(final["generation"]["max_out_length"], 10240)
            shard = final_config(selected_path, "c2")
            self.assertEqual(
                [profile["id"] for profile in shard["system_prompts"]["profiles"]],
                ["C2"],
            )
            self.assertEqual(Path(shard["execution"]["output_dir"]).name, "c2")

            length = final_config(
                selected_path,
                "c1",
                max_assistant_tokens=16384,
                max_trajectory_tokens=24576,
                experiment_tag="l16k-t24k",
                execution_timeout_seconds=600,
            )
            self.assertEqual(length["generation"]["max_out_length"], 16384)
            self.assertEqual(
                length["benchmarks"][0]["options"]["max_trajectory_tokens"],
                24576,
            )
            self.assertEqual(length["benchmarks"][0]["limit"], 200)
            self.assertEqual(
                Path(length["execution"]["output_dir"]).parts[-2:],
                ("evaluation-l16k-t24k", "c1"),
            )
            self.assertEqual(
                length["experiment"]["name"],
                "retool-coding-0812-evaluation-l16k-t24k-c1-seed42",
            )
            self.assertEqual(
                {model["id"] for model in length["models"]},
                {"base", "c0", "c1", "c2"},
            )
            self.assertEqual(length["tracking"]["job_type"], "eval-l16k-t24k-c1")
            self.assertEqual(length["execution"]["timeout_seconds"], 600)

            canary = final_config(
                selected_path,
                "c0",
                max_assistant_tokens=16384,
                max_trajectory_tokens=24576,
                experiment_tag="l16k-t24k-canary",
                tasks=1,
            )
            self.assertEqual(canary["benchmarks"][0]["limit"], 1)
            self.assertLessEqual(len(canary["tracking"]["job_type"]), 20)

    def test_length_ablation_rejects_unsafe_overrides(self) -> None:
        with TemporaryDirectory() as raw:
            path = Path(raw) / "selected.json"
            path.write_text(
                json.dumps(
                    {
                        "base": None,
                        **{
                            variant: {
                                "model_path": f"trio://{variant}/weights",
                                "train_variant": variant,
                                "step": 100,
                                "seed": 42,
                            }
                            for variant in ("c0", "c1", "c2")
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Invalid experiment tag"):
                final_config(path, "c0", experiment_tag="../bad")
            with self.assertRaisesRegex(ValueError, "at least max_assistant_tokens"):
                final_config(
                    path,
                    "c0",
                    max_assistant_tokens=16384,
                    max_trajectory_tokens=12000,
                )


if __name__ == "__main__":
    unittest.main()
