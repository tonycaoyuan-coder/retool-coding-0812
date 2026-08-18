"""Train one frozen C0/C1/C2 formal branch with PyTRIO GRPO."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from time import perf_counter
from typing import Any

import pytrio as trio
import swanlab
from tqdm import tqdm

from retool_coding_0812.data import file_sha256, load_examples, stable_hash, take_training_batch
from retool_coding_0812.docker_executor import (
    DockerConfig,
    DockerExecutor,
    DockerInfrastructureError,
)
from retool_coding_0812.protocol import TOOLS, system_prompt
from retool_coding_0812.resume import (
    latest_checkpoint,
    load_checkpoint,
    read_metric_rows,
    reconcile_local_artifacts,
    validate_metric_prefix,
)
from retool_coding_0812.rollout import RolloutConfig, rollout_batch
from retool_coding_0812.telemetry import save_trajectory_gzip
from retool_coding_0812.gates import (
    experiment_fingerprint,
    load_json,
    protocol_payload,
    validate_data_manifest,
    validate_preflight,
    validate_selected_protocol,
    validate_training_gate,
)
from retool_coding_0812.training_utils import (
    build_training_datums,
    merge_trainer_metrics,
    pack_micro_batches,
    early_training_signal_gate,
    rollout_metrics,
    weight_micro_batch_for_global_mean,
)


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def save_checkpoint(
    client: Any,
    *,
    output_dir: Path,
    run_name: str,
    step: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    name = f"{run_name}-step-{step}"
    state = client.save_state(name=f"{name}-state").result()
    weights = client.save_weights_for_sampler(name=f"{name}-weights").result()
    record = {
        **metadata,
        "name": name,
        "step": step,
        "state_path": state.path,
        "sampler_weights_path": weights.path,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.json"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint record {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return record


def _training_client(
    service: Any,
    args: argparse.Namespace,
    resume: dict[str, Any] | None,
    initial: dict[str, Any] | None,
) -> Any:
    if resume:
        return service.create_training_client_from_state_with_optimizer(str(resume["state_path"]))
    if initial:
        # Shared SFT supplies the identical LoRA initialization for C0/C1/C2,
        # but its cross-entropy Adam moments must not leak into GRPO.
        return service.create_training_client_from_state(str(initial["state_path"]))
    return service.create_lora_training_client(
        base_model=args.base_model,
        rank=args.lora_rank,
        seed=args.seed,
    )


def _trainer_sequence_limit(service: Any, base_model: str, fallback: int) -> int:
    capabilities = service.get_server_capabilities()
    for model in capabilities.supported_models:
        if model.model_name == base_model:
            if not model.training.available:
                raise ValueError(f"Training is unavailable for {base_model}")
            advertised = model.training.max_seq_len
            return min(fallback, int(advertised)) if advertised is not None else fallback
    raise ValueError(f"Training capabilities are unavailable for {base_model}")


def _initial_state_provenance(
    initial: dict[str, Any] | None,
    resume: dict[str, Any] | None,
    initial_manifest_path: Path | None,
) -> tuple[str, str | None]:
    if initial:
        if initial_manifest_path is None:
            raise ValueError("Initial state manifest path is required")
        return str(initial["recipe_fingerprint"]), file_sha256(initial_manifest_path)
    if resume:
        recipe_fingerprint = resume.get("initial_state_recipe_fingerprint")
        manifest_sha256 = resume.get("initial_state_manifest_sha256")
        if not recipe_fingerprint or not manifest_sha256:
            raise ValueError("Resume checkpoint is missing initial-state provenance")
        return str(recipe_fingerprint), str(manifest_sha256)
    return "base", None


def validate_resume(
    resume: dict[str, Any],
    invariants: dict[str, Any],
    run_dir: Path,
    checkpoint_dir: Path,
    start_step: int,
) -> list[dict[str, Any]]:
    """Verify that local evidence and the remote checkpoint describe one run.

    Anything produced after the last saved optimizer state is non-authoritative
    and is moved into ``recovery/`` before deterministic replay.
    """

    mismatches = {
        key: (resume.get(key), expected)
        for key, expected in invariants.items()
        if resume.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Resume checkpoint invariants differ: {mismatches}")
    rows = validate_metric_prefix(
        read_metric_rows(run_dir / "metrics.jsonl"), start_step
    )
    trajectory_dir = run_dir / "trajectories" / f"step-{start_step:04d}"
    expected_trajectories = int(invariants["questions_per_batch"]) * int(
        invariants["group_size"]
    )
    if len(list(trajectory_dir.glob("*.json.gz"))) != expected_trajectories:
        raise ValueError(
            f"Resume step must have exactly {expected_trajectories} authoritative "
            f"trajectories: {trajectory_dir}"
        )
    if start_step >= int(invariants["early_gate_step"]):
        gate_path = run_dir / "early-gate.json"
        gate_rows = rows[: int(invariants["early_gate_step"])]
        gate = early_training_signal_gate(
            gate_rows,
            min_nondegenerate_group_rate=float(
                invariants["min_early_nondegenerate_rate"]
            ),
            max_skipped_update_rate=float(
                invariants["max_early_skipped_update_rate"]
            ),
        )
        if not gate["passed"]:
            raise ValueError(f"Resume checkpoint failed the early-training gate: {gate}")
        if gate_path.exists():
            recorded_gate = json.loads(gate_path.read_text(encoding="utf-8"))
            if recorded_gate != gate:
                raise ValueError(f"Recorded early-training gate differs: {gate_path}")
        else:
            gate_path.write_text(
                json.dumps(gate, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    return reconcile_local_artifacts(
        run_dir=run_dir,
        checkpoint_step=start_step,
        checkpoint_dir=checkpoint_dir,
        checkpoint_pattern=f"{invariants['run_name']}-step-*.json",
        expected_run_name=str(invariants["run_name"]),
    )


def main(args: argparse.Namespace) -> None:
    """Execute one frozen GRPO branch from gates through final checkpoint.

    A step samples a complete same-prompt group, computes group-relative
    advantages, accumulates weighted micro-batches, then performs one optimizer
    update. Degenerate groups remain in telemetry but carry no update signal.
    """

    if min(
        args.max_steps,
        args.questions_per_batch,
        args.group_size,
        args.save_every,
        args.early_gate_step,
    ) < 1:
        raise ValueError("steps, batch size, group size, and save interval must be positive")
    if args.max_step_retries < 0:
        raise ValueError("max-step-retries must be non-negative")
    examples = load_examples(args.data)
    if args.max_steps * args.questions_per_batch > len(examples):
        raise ValueError("The default experiment must not repeat training tasks")
    if args.max_steps == 100 and args.questions_per_batch == 4 and len(examples) != 400:
        raise ValueError("The formal schedule requires the exact frozen 400-task train split")
    resume = load_checkpoint(args.resume_checkpoint) if args.resume_checkpoint else None
    initial = load_json(args.initial_state_manifest) if args.initial_state_manifest else None
    if initial and (
        not initial.get("complete")
        or initial.get("base_model") != args.base_model
        or int(initial.get("seed", -1)) != args.seed
        or not initial.get("state_path")
        or not initial.get("recipe_fingerprint")
    ):
        raise ValueError("Initial shared-SFT state manifest is incomplete or mismatched")
    if resume and initial:
        raise ValueError("Use resume-checkpoint or initial-state-manifest, not both")
    start_step = int(resume.get("step", 0)) if resume else 0
    if start_step > args.max_steps:
        raise ValueError("Resume checkpoint step is beyond max-steps")
    run_name = args.run_name or (resume.get("run_name") if resume else None) or (
        f"retool-coding-0812-{args.prompt_variant}-seed{args.seed}"
    )
    run_dir = args.artifact_dir / run_name
    if not args.gate_only and not resume and not args.recover:
        existing_checkpoint = latest_checkpoint(
            args.checkpoint_dir,
            pattern=f"{run_name}-step-*.json",
            expected_run_name=run_name,
        )
        if existing_checkpoint is not None or read_metric_rows(run_dir / "metrics.jsonl"):
            raise FileExistsError(
                f"Partial or completed training artifacts exist for {run_name}; "
                "rerun with --resume"
            )
    executor = DockerExecutor(
        DockerConfig(
            docker_binary=args.docker_binary,
            image=args.image,
            max_workers=args.sandbox_workers,
        )
    )
    image_id = executor.check_runtime()
    validate_preflight(args.preflight_manifest, image=args.image, image_id=image_id)
    data_gate = validate_data_manifest(args.data, args.data_manifest)
    protocol = protocol_payload(vars(args))
    model_fingerprint, initial_state_manifest_sha256 = _initial_state_provenance(
        initial, resume, args.initial_state_manifest
    )
    experiment_id, fingerprint_payload = experiment_fingerprint(
        base_model=args.base_model,
        dataset_manifest_sha256=data_gate["manifest_sha256"],
        protocol=protocol,
        model_fingerprint=model_fingerprint,
    )
    validate_training_gate(
        args.smoke_manifest,
        base_model=args.base_model,
        dataset_manifest_sha256=data_gate["manifest_sha256"],
        protocol=protocol,
        prompt_variant=args.prompt_variant,
        model_fingerprint=model_fingerprint,
    )
    validate_selected_protocol(
        args.protocol_config,
        smoke_path=args.smoke_manifest,
        base_model=args.base_model,
        dataset_manifest_sha256=data_gate["manifest_sha256"],
        protocol=protocol,
        model_fingerprint=model_fingerprint,
    )
    invariants = {
        "run_name": run_name,
        "prompt_variant": args.prompt_variant,
        "base_model": args.base_model,
        "seed": args.seed,
        "data_sha256": file_sha256(args.data),
        "system_prompt_sha256": stable_hash(
            system_prompt(args.prompt_variant, max_tool_calls=args.max_tool_calls)
        ),
        "tools_sha256": stable_hash(TOOLS),
        "docker_image": args.image,
        "docker_image_id": image_id,
        "lora_rank": args.lora_rank,
        "learning_rate": args.learning_rate,
        "adam_beta1": args.beta1,
        "adam_beta2": args.beta2,
        "questions_per_batch": args.questions_per_batch,
        "group_size": args.group_size,
        "max_steps": args.max_steps,
        "save_every": args.save_every,
        "early_gate_step": args.early_gate_step,
        "min_early_nondegenerate_rate": args.min_early_nondegenerate_rate,
        "max_early_skipped_update_rate": args.max_early_skipped_update_rate,
        "max_tool_calls": args.max_tool_calls,
        "max_assistant_turns": args.max_assistant_turns,
        "max_trajectory_tokens": args.max_trajectory_tokens,
        "max_assistant_tokens": args.max_assistant_tokens,
        "max_tool_response_tokens": args.max_tool_response_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "preflight_manifest_sha256": file_sha256(args.preflight_manifest),
        "data_manifest_sha256": data_gate["manifest_sha256"],
        "experiment_fingerprint": experiment_id,
        "fingerprint_payload": fingerprint_payload,
        "initial_state_recipe_fingerprint": model_fingerprint,
        "initial_state_manifest_sha256": initial_state_manifest_sha256,
    }
    committed_rows: list[dict[str, Any]] = []
    if resume:
        committed_rows = validate_resume(
            resume, invariants, run_dir, args.checkpoint_dir, start_step
        )
    elif args.recover:
        committed_rows = reconcile_local_artifacts(
            run_dir=run_dir,
            checkpoint_step=0,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_pattern=f"{run_name}-step-*.json",
            expected_run_name=run_name,
        )
    if args.gate_only:
        print(
            json.dumps(
                {
                    "gate_only": True,
                    "experiment_fingerprint": experiment_id,
                    "run_name": run_name,
                    "prompt_variant": args.prompt_variant,
                    "seed": args.seed,
                    "protocol": protocol,
                    "model_fingerprint": model_fingerprint,
                },
                indent=2,
            )
        )
        return
    if start_step == args.max_steps:
        print(
            json.dumps(
                {
                    "complete": True,
                    "run_name": run_name,
                    "step": start_step,
                    "checkpoint": str(args.resume_checkpoint),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    service = trio.ServiceClient()
    training_client = _training_client(service, args, resume, initial)
    tokenizer = training_client.get_tokenizer()
    max_training_context_tokens = _trainer_sequence_limit(
        service,
        args.base_model,
        args.max_trajectory_tokens,
    )
    init_kwargs: dict[str, Any] = {
        "project": args.swanlab_project,
        "name": run_name,
        "mode": args.swanlab_mode,
        "config": {**serializable_args(args), **invariants},
        "group": args.swanlab_group,
        "job_type": f"train-{args.prompt_variant.upper()}",
        "tags": ["formal", "0812", args.prompt_variant.upper(), f"seed-{args.seed}"],
        "log_dir": str(run_dir / "swanlog"),
    }
    if resume and resume.get("swanlab_run_id"):
        init_kwargs.update(resume="must", id=resume["swanlab_run_id"])
    run = swanlab.init(**init_kwargs)
    adam = trio.AdamParams(
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
    )
    reward_window: deque[float] = deque(
        (
            float(row["reward/mean"])
            for row in committed_rows[-10:]
            if "reward/mean" in row
        ),
        maxlen=10,
    )
    early_gate_rows: list[dict[str, float]] = [
        row for row in committed_rows if int(row["step"]) <= args.early_gate_step
    ]
    try:
        with tqdm(total=args.max_steps - start_step, desc=f"Train {args.prompt_variant}") as progress:
            for step in range(start_step, args.max_steps):
                started = perf_counter()
                docker_before = executor.stats.snapshot()
                batch = take_training_batch(
                    examples,
                    step=step,
                    questions_per_batch=args.questions_per_batch,
                )
                sampler = training_client.save_weights_and_get_sampling_client()
                rollout_config = RolloutConfig(
                    prompt_variant=args.prompt_variant,
                    group_size=args.group_size,
                    max_tool_calls=args.max_tool_calls,
                    max_assistant_turns=args.max_assistant_turns,
                    max_trajectory_tokens=args.max_trajectory_tokens,
                    max_assistant_tokens=args.max_assistant_tokens,
                    max_tool_response_tokens=args.max_tool_response_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=args.seed + step * 10_000,
                )
                for attempt in range(args.max_step_retries + 1):
                    try:
                        trajectories = rollout_batch(
                            sampler,
                            tokenizer,
                            executor,
                            batch,
                            rollout_config,
                        )
                        break
                    except DockerInfrastructureError:
                        if attempt == args.max_step_retries:
                            raise
                for trajectory in trajectories:
                    save_trajectory_gzip(
                        run_dir
                        / "trajectories"
                        / f"step-{step + 1:04d}"
                        / f"{trajectory.example.instance_id}-g{trajectory.group_index}.json.gz",
                        trajectory,
                        metadata={**invariants, "step": step + 1},
                    )
                datums = build_training_datums(
                    trajectories,
                    max_context_tokens=max_training_context_tokens,
                )
                datum_candidates = sum(
                    item.advantage != 0.0 and bool(item.turns) for item in trajectories
                )
                overlength_datums = datum_candidates - len(datums)
                micro_batches = pack_micro_batches(
                    datums,
                    max_items=args.max_micro_batch_items,
                    max_padded_tokens=args.max_micro_batch_padded_tokens,
                )
                trainer_results = []
                for micro_batch in micro_batches:
                    trainer_results.append(
                        training_client.forward_backward(
                            weight_micro_batch_for_global_mean(micro_batch, len(trajectories)),
                            loss_fn="importance_sampling",
                        ).result()
                    )
                if micro_batches:
                    training_client.optim_step(adam).result()
                metrics = rollout_metrics(
                    trajectories,
                    datums,
                    micro_batches,
                    question_count=len(batch),
                )
                metrics["train/max_context_tokens"] = float(max_training_context_tokens)
                metrics["train/overlength_datums"] = float(overlength_datums)
                metrics["train/overlength_datum_rate"] = (
                    overlength_datums / datum_candidates if datum_candidates else 0.0
                )
                metrics.update(executor.stats.metrics_since(docker_before))
                metrics.update(merge_trainer_metrics(trainer_results))
                metrics["time/step_seconds"] = perf_counter() - started
                reward_window.append(metrics["reward/mean"])
                metrics["reward/mean_ma10"] = sum(reward_window) / len(reward_window)
                if step + 1 <= args.early_gate_step:
                    early_gate_rows.append(metrics)
                run_dir.mkdir(parents=True, exist_ok=True)
                with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"step": step + 1, **metrics}) + "\n")
                swanlab.log(metrics, step=step + 1)
                if (step + 1) % args.save_every == 0 or step + 1 == args.max_steps:
                    save_checkpoint(
                        training_client,
                        output_dir=args.checkpoint_dir,
                        run_name=run_name,
                        step=step + 1,
                        metadata={
                            **invariants,
                            "swanlab_run_id": getattr(run, "id", None),
                        },
                    )
                if step + 1 == args.early_gate_step:
                    gate = early_training_signal_gate(
                        early_gate_rows,
                        min_nondegenerate_group_rate=args.min_early_nondegenerate_rate,
                        max_skipped_update_rate=args.max_early_skipped_update_rate,
                    )
                    (run_dir / "early-gate.json").write_text(
                        json.dumps(gate, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    if not gate["passed"]:
                        raise RuntimeError(
                            "Early training signal gate failed at step "
                            f"{args.early_gate_step}: {gate}"
                        )
                progress.update(1)
                progress.set_postfix(
                    reward=f"{metrics['reward/mean']:.3f}",
                    pass1=f"{metrics['reward/pass_at_1']:.3f}",
                    degenerate=f"{metrics['rollout/degenerate_group_rate']:.2f}",
                )
    except KeyboardInterrupt:
        swanlab.finish(state="aborted")
        raise
    except Exception as exc:
        swanlab.finish(state="crashed", error=str(exc))
        raise
    else:
        swanlab.finish()
