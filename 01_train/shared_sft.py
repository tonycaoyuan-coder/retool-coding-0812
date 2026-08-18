"""Reproduce the frozen three-epoch seed-42 shared-neutral SFT initialization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import pytrio as trio
import swanlab

from retool_coding_0812.data import file_sha256, stable_hash
from retool_coding_0812.resume import (
    latest_checkpoint,
    load_checkpoint,
    read_metric_rows,
    reconcile_local_artifacts,
)
from retool_coding_0812.settings import ROOT, load_config, resolve_path
from retool_coding_0812.sft_utils import load_validated_sft_datums
from retool_coding_0812.training_utils import merge_trainer_metrics, pack_micro_batches


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _save_checkpoint(
    client: Any,
    *,
    output_dir: Path,
    step: int,
    recipe_fingerprint: str,
    swanlab_run_id: str | None,
) -> None:
    name = f"retool-coding-0812-shared-sft-seed42-step-{step}"
    state = client.save_state(name=f"{name}-state").result()
    destination = output_dir / "checkpoints" / f"step-{step}.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite SFT checkpoint {destination}")
    _write_json(
        destination,
        {
            "name": name,
            "step": step,
            "state_path": state.path,
            "recipe_fingerprint": recipe_fingerprint,
            "swanlab_run_id": swanlab_run_id,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume-checkpoint", type=Path)
    resume.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest shared-SFT checkpoint, or restart safely at step 0.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    """Run or resume the frozen SFT schedule and publish its final state manifest.

    The saved state includes optimizer moments for SFT resume. GRPO branches
    intentionally load the resulting model state without those moments.
    """

    args = args or parse_args()
    config = load_config()
    experiment = config["experiment"]
    inputs = config["inputs"]
    settings = config["shared_sft"]
    tracking = config["swanlab"]
    source_manifest = resolve_path(inputs["shared_sft_source_manifest"])
    output_dir = ROOT / "artifacts/training/shared-sft/seed42"
    final_manifest = output_dir / "manifest.json"
    datums, source = load_validated_sft_datums(
        source_manifest,
        max_context_tokens=settings["max_context_tokens"],
    )
    micro_batches = pack_micro_batches(
        datums,
        max_items=settings["max_micro_batch_items"],
        max_padded_tokens=settings["max_micro_batch_padded_tokens"],
    )
    schedule = micro_batches * settings["epochs"]
    recipe = {
        "kind": "shared-neutral-cold-start-sft-v1",
        "base_model": experiment["base_model"],
        "source_manifest_sha256": file_sha256(source_manifest),
        "lora_rank": experiment["lora_rank"],
        "epochs": settings["epochs"],
        "learning_rate": settings["learning_rate"],
        "beta1": settings["beta1"],
        "beta2": settings["beta2"],
        "max_context_tokens": settings["max_context_tokens"],
        "max_micro_batch_items": settings["max_micro_batch_items"],
        "max_micro_batch_padded_tokens": settings["max_micro_batch_padded_tokens"],
        "loss_fn": "cross_entropy",
        "mask": "assistant-only manual autoregressive shift",
    }
    recipe_fingerprint = stable_hash(recipe)
    if recipe_fingerprint != "6bf17aed47f7be99b58c19ede75ab016b73c17213768d60322642e3a224b0e1d":
        raise ValueError(f"Shared-SFT recipe fingerprint drifted: {recipe_fingerprint}")
    if final_manifest.exists():
        completed = json.loads(final_manifest.read_text(encoding="utf-8"))
        if not isinstance(completed, dict):
            raise ValueError(f"Completed shared-SFT manifest is invalid: {final_manifest}")
        expected = {
            "complete": True,
            "base_model": experiment["base_model"],
            "seed": experiment["seed"],
            "lora_rank": experiment["lora_rank"],
            "recipe_fingerprint": recipe_fingerprint,
            "source_fingerprint": source["source_fingerprint"],
            "source_manifest_sha256": file_sha256(source_manifest),
            "optimizer_steps": len(schedule),
        }
        mismatches = {
            key: (completed.get(key), value)
            for key, value in expected.items()
            if completed.get(key) != value
        }
        if mismatches or not completed.get("state_path") or not completed.get(
            "sampler_weights_path"
        ):
            raise ValueError(f"Completed shared-SFT manifest is invalid: {mismatches}")
        if args.resume:
            print(final_manifest)
            return
        raise FileExistsError(f"Refusing to overwrite completed SFT: {final_manifest}")

    checkpoint_dir = output_dir / "checkpoints"
    resume_checkpoint = args.resume_checkpoint
    if args.resume and resume_checkpoint is None:
        resume_checkpoint = latest_checkpoint(checkpoint_dir, pattern="step-*.json")
    resume = load_checkpoint(resume_checkpoint) if resume_checkpoint is not None else None
    start_step = int(resume["step"]) if resume else 0
    if start_step >= len(schedule):
        raise ValueError(
            f"Shared-SFT checkpoint step {start_step} is beyond the resumable schedule"
        )
    if resume and resume.get("recipe_fingerprint") != recipe_fingerprint:
        raise ValueError("Shared-SFT checkpoint recipe fingerprint differs")
    if not resume and not args.resume and latest_checkpoint(checkpoint_dir, pattern="step-*.json"):
        raise FileExistsError("Shared-SFT checkpoints exist; rerun with --resume")
    if not resume and not args.resume and read_metric_rows(output_dir / "metrics.jsonl"):
        raise FileExistsError("Partial shared-SFT metrics exist; rerun with --resume")
    reconcile_local_artifacts(
        run_dir=output_dir,
        checkpoint_step=start_step,
        checkpoint_dir=checkpoint_dir,
        checkpoint_pattern="step-*.json",
        trajectory_dir_name=None,
    )

    service = trio.ServiceClient()
    client = (
        service.create_training_client_from_state_with_optimizer(str(resume["state_path"]))
        if resume
        else service.create_lora_training_client(
            base_model=experiment["base_model"],
            rank=experiment["lora_rank"],
            seed=experiment["seed"],
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    init_kwargs: dict[str, Any] = dict(
        project=tracking["project"],
        name="retool-coding-0812-shared-sft-seed42",
        mode=tracking["mode"],
        config={
            **recipe,
            "recipe_fingerprint": recipe_fingerprint,
            "seed": experiment["seed"],
            "source_fingerprint": source["source_fingerprint"],
            "source_selected": len(source["selected"]),
            "optimizer_steps": len(schedule),
            "resume_step": start_step,
        },
        group=tracking["group"],
        job_type="shared-sft",
        tags=["formal", "0812", "shared-sft", "seed-42"],
        log_dir=str(output_dir / "swanlog"),
    )
    if resume and resume.get("swanlab_run_id"):
        init_kwargs.update(resume="must", id=resume["swanlab_run_id"])
    run = swanlab.init(**init_kwargs)
    adam = trio.AdamParams(
        learning_rate=settings["learning_rate"],
        beta1=settings["beta1"],
        beta2=settings["beta2"],
    )
    metrics_path = output_dir / "metrics.jsonl"
    started_all = perf_counter()
    try:
        for step, batch in enumerate(schedule[start_step:], start=start_step + 1):
            started = perf_counter()
            result = client.forward_backward(
                [item.datum for item in batch], loss_fn="cross_entropy"
            ).result()
            client.optim_step(adam).result()
            metrics = merge_trainer_metrics([result])
            metrics.update(
                {
                    "sft/epoch": (step - 1) // len(micro_batches) + 1,
                    "sft/micro_batch_items": len(batch),
                    "sft/padded_tokens": max(item.num_tokens for item in batch) * len(batch),
                    "sft/optimizer_step": step,
                    "time/step_seconds": perf_counter() - started,
                }
            )
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"step": step, **metrics}) + "\n")
            swanlab.log(metrics, step=step)
            print(json.dumps({"step": step, "total": len(schedule), **metrics}), flush=True)
            if step % settings["save_every"] == 0 and step != len(schedule):
                _save_checkpoint(
                    client,
                    output_dir=output_dir,
                    step=step,
                    recipe_fingerprint=recipe_fingerprint,
                    swanlab_run_id=getattr(run, "id", None),
                )
        state = client.save_state(
            name="retool-coding-0812-shared-sft-seed42-final-state"
        ).result()
        weights = client.save_weights_for_sampler(
            name="retool-coding-0812-shared-sft-seed42-final-weights"
        ).result()
        manifest = {
            "complete": True,
            "name": "retool-coding-0812-shared-sft-seed42",
            "base_model": experiment["base_model"],
            "seed": experiment["seed"],
            "lora_rank": experiment["lora_rank"],
            "recipe_fingerprint": recipe_fingerprint,
            "recipe": recipe,
            "source_fingerprint": source["source_fingerprint"],
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": file_sha256(source_manifest),
            "selected_trajectories": len(source["selected"]),
            "optimizer_steps": len(schedule),
            "state_path": state.path,
            "sampler_weights_path": weights.path,
            "swanlab_run_id": getattr(run, "id", None),
            "duration_seconds": perf_counter() - started_all,
            "metrics_path": "metrics.jsonl",
        }
        _write_json(final_manifest, manifest)
    except KeyboardInterrupt:
        swanlab.finish(state="aborted")
        raise
    except Exception as exc:
        swanlab.finish(state="crashed", error=str(exc))
        raise
    else:
        swanlab.finish()
        print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
