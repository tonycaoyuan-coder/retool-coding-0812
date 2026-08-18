"""Run and summarize the C0-step100/shared-SFT-only post-hoc evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any

from myeval.cli import app
from myeval.config import load_config as load_myeval_config
from myeval.tracking import sync_swanlab
import yaml

import retool_coding_0812.myeval_plugin  # noqa: F401
from retool_coding_0812.eval_config import _base_config, _model_entry, write_yaml
from retool_coding_0812.eval_resume import evaluation_run_lock, resume_mode
from retool_coding_0812.parallel import run_parallel
from retool_coding_0812.settings import ROOT, load_config, resolve_path


VARIANTS = ("c0", "c1", "c2")
CONFIG_DIR = ROOT / "configs/generated/evaluation-posthoc"
OUTPUT_ROOT = ROOT / "artifacts/evaluation-posthoc"
MODEL_MANIFEST = OUTPUT_ROOT / "models.json"
REPORT_PATH = ROOT / "docs/results/posthoc-checkpoint-and-sft-ablation.md"
EXPERIMENT_PATH = ROOT / "docs/experiment-record-and-results.md"
SECTION_MARKER = "## 2026-08-13 — Post-hoc evaluation: C0-step100 与 shared-SFT-only"
SWANLAB_REPORT_METRICS = (
    "case_pass_rate",
    "compile_error_rate",
    "error_rate",
    "format_valid_rate",
    "latency_seconds_mean",
    "mean_execution_seconds",
    "mean_tool_calls",
    "mean_trajectory_tokens",
    "mean_turns",
    "pass@1",
    "pass_at_1",
    "private_pass_rate",
    "public_pass_rate",
    "runtime_error_rate",
    "time_limit_rate",
    "token_cap_hit_rate",
    "tool_call_valid_rate",
    "tool_use_rate",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare() -> None:
    checkpoints = _load_json(ROOT / "artifacts/training/checkpoints/models.json")
    shared = _load_json(ROOT / "artifacts/training/shared-sft/seed42/manifest.json")
    models = {
        "c0-step100": checkpoints["c0-step-100"],
        "shared-sft-only": {
            "model_path": shared["sampler_weights_path"],
            "train_variant": "shared-sft-only",
            "step": int(shared["optimizer_steps"]),
            "seed": int(shared["seed"]),
        },
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_MANIFEST.write_text(json.dumps(models, indent=2) + "\n", encoding="utf-8")
    entries = [
        _model_entry("c0-step100", models["c0-step100"], "evaluation-posthoc"),
        _model_entry("shared-sft-only", models["shared-sft-only"], "evaluation-posthoc"),
    ]
    config = load_config()
    for variant in VARIANTS:
        value = _base_config(
            name=f"retool-coding-0812-posthoc-{variant}-seed42",
            data=resolve_path(f"{config['inputs']['formal_data_dir']}/test.jsonl.gz"),
            tasks=int(config["evaluation"]["tasks"]),
            models=entries,
            role=f"posthoc-{variant}",
            output_dir=OUTPUT_ROOT / variant,
            prompt_variants=(variant,),
        )
        value["experiment"]["description"] = (
            "Post-hoc checkpoint/SFT ablation on the frozen seed-42 0812 test set; "
            "does not replace selected-checkpoint final evaluation."
        )
        value["tracking"]["group"] = "retool-coding-0812-posthoc-seed42"
        write_yaml(CONFIG_DIR / f"{variant}.yaml", value)


def _run(variant: str | None, *, validate_only: bool, resume: bool) -> None:
    prepare()
    if variant is None:
        commands: list[list[str]] = []
        labels: list[str] = []
        for item in VARIANTS:
            config = CONFIG_DIR / f"{item}.yaml"
            mode = "run" if validate_only else resume_mode(config, resume)
            if mode == "skip":
                print(f"[posthoc] skipping completed {item}", flush=True)
                continue
            command = [sys.executable, str(Path(__file__).resolve()), "run", "--variant", item]
            if validate_only:
                command.append("--validate-only")
            if mode == "resume":
                command.append("--resume")
            commands.append(command)
            labels.append(item)
        if commands:
            run_parallel(commands, cwd=ROOT, labels=labels)
        return
    config = CONFIG_DIR / f"{variant}.yaml"
    mode = "run" if validate_only else resume_mode(config, resume)
    if mode == "skip":
        print(f"[posthoc] skipping completed {variant}", flush=True)
        return
    with evaluation_run_lock(config):
        sys.argv = ["myeval", "validate" if validate_only else "run", str(config)]
        if mode == "resume" and not validate_only:
            sys.argv.append("--resume")
        app()


def _complete_run(config_path: Path, expected: int) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = Path(config["execution"]["output_dir"])
    candidates = []
    for manifest_path in output_dir.glob("*/manifest.json"):
        manifest = _load_json(manifest_path)
        if manifest.get("status") == "complete":
            candidates.append(manifest_path.parent)
    if not candidates:
        raise RuntimeError(f"No complete run for {config_path}")
    run_dir = max(candidates, key=lambda path: path.stat().st_mtime)
    summary = _load_json(run_dir / "summary/metrics.json")
    counts = summary.get("counts") or {}
    if counts.get("completed") != expected or any(counts.get(key, 0) for key in ("pending", "running", "failed")):
        raise RuntimeError(f"Incomplete post-hoc shard {run_dir}: {counts}")
    return run_dir


def _cells(run_dir: Path) -> list[dict[str, Any]]:
    return list((_load_json(run_dir / "summary/metrics.json").get("matrix_analysis") or {}).get("cells") or [])


def _sample_values(run_dir: Path) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], float] = {}
    for score_path in sorted((run_dir / "scores").glob("*.jsonl")):
        for line in score_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("status") != "completed":
                continue
            score = record.get("score") or {}
            values[(str(record["model_id"]), str(record["sample_id"]))] = float(
                (score.get("metrics") or {}).get("pass_at_1", 0.0)
            )
    return values


def _paired_delta(a: dict[str, float], b: dict[str, float]) -> tuple[float, int, int, int]:
    common = sorted(set(a) & set(b))
    diffs = [a[key] - b[key] for key in common]
    wins = sum(value > 0 for value in diffs)
    losses = sum(value < 0 for value in diffs)
    return statistics.fmean(diffs), wins, len(common) - wins - losses, losses


def finalize() -> None:
    runs = {variant: _complete_run(CONFIG_DIR / f"{variant}.yaml", 400) for variant in VARIANTS}
    posthoc_cells: dict[tuple[str, str], dict[str, Any]] = {}
    for variant, run_dir in runs.items():
        for cell in _cells(run_dir):
            posthoc_cells[(str(cell["model_id"]), variant)] = cell
    original_runs = {
        variant: _complete_run(ROOT / f"configs/generated/evaluation/{variant}.yaml", 800)
        for variant in VARIANTS
    }
    original_cells: dict[tuple[str, str], dict[str, Any]] = {}
    comparisons: dict[tuple[str, str], tuple[float, int, int, int]] = {}
    for variant in VARIANTS:
        for cell in _cells(original_runs[variant]):
            original_cells[(str(cell["model_id"]), variant)] = cell
        new_values = _sample_values(runs[variant])
        old_values = _sample_values(original_runs[variant])
        for new_model, old_model in (("c0-step100", "c0"), ("shared-sft-only", "base")):
            new = {sample: value for (model, sample), value in new_values.items() if model == new_model}
            old = {sample: value for (model, sample), value in old_values.items() if model == old_model}
            comparisons[(new_model, variant)] = _paired_delta(new, old)

    def values(model: str) -> list[float]:
        return [float(posthoc_cells[(model, variant)]["value"]) for variant in VARIANTS]

    lines = [
        "# ReTool-Coding-0812 Post-hoc Evaluation",
        "",
        "> Seed 42; the original 200 temporally held-out tasks; greedy decoding. "
        "This is a post-hoc checkpoint/SFT ablation and does not replace the preregistered selected-checkpoint final.",
        "",
        "| Model | Test C0 | Test C1 | Test C2 | Average | Worst |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in ("c0-step100", "shared-sft-only"):
        row = values(model)
        lines.append(
            f"| {model} | {row[0]:.3f} | {row[1]:.3f} | {row[2]:.3f} | "
            f"{statistics.fmean(row):.3f} | {min(row):.3f} |"
        )
    lines += [
        "",
        "## Paired comparisons on the same 200 tasks",
        "",
        "| Contrast | Test | Delta | Wins/Ties/Losses |",
        "|---|---|---:|---:|",
    ]
    for model, reference in (("c0-step100", "selected C0-step40"), ("shared-sft-only", "raw Base")):
        for variant in VARIANTS:
            delta, wins, ties, losses = comparisons[(model, variant)]
            lines.append(f"| {model} - {reference} | {variant.upper()} | {delta:+.3f} | {wins}/{ties}/{losses} |")
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "- C0-step100 isolates the checkpoint-step sensitivity of the C0 branch; it is not a replacement checkpoint selected after seeing final results.",
        "- shared-SFT-only separates the common neutral SFT contribution from the subsequent prompt-conditioned GRPO branches.",
        "- Both additions are post-hoc and reuse the original test set, so they refine mechanism attribution rather than constitute a new confirmatory test.",
        "",
        "## Provenance",
        "",
    ]
    for variant in VARIANTS:
        lines.append(f"- {variant.upper()}: `{runs[variant].relative_to(ROOT)}`")
    report = "\n".join(lines) + "\n"
    REPORT_PATH.write_text(report, encoding="utf-8")

    experiment = EXPERIMENT_PATH.read_text(encoding="utf-8")
    section = SECTION_MARKER + "\n\n" + "\n".join(lines[2:]) + "\n"
    if SECTION_MARKER in experiment:
        experiment = experiment[: experiment.index(SECTION_MARKER)].rstrip() + "\n\n" + section
    else:
        experiment = experiment.rstrip() + "\n\n" + section
    EXPERIMENT_PATH.write_text(experiment, encoding="utf-8")
    print(REPORT_PATH)
    print(EXPERIMENT_PATH)


def upload_swanlab(variant: str | None = None) -> None:
    """Upload the three completed local summaries with SwanLab-safe job names."""

    prepare()
    selected = (variant,) if variant is not None else VARIANTS
    for item in selected:
        config_path = CONFIG_DIR / f"{item}.yaml"
        run_dir = _complete_run(config_path, 400)
        summary = _load_json(run_dir / "summary/metrics.json")
        config = load_myeval_config(config_path)
        sync_swanlab(config, run_dir, summary)
        print(f"uploaded {item}: {run_dir}", flush=True)


def upload_swanlab_model(model_id: str) -> None:
    """Upload one model-centric run containing all three test prompts."""

    import swanlab
    from pyecharts import options as opts
    from pyecharts.charts import HeatMap

    prepare()
    rows: dict[str, dict[str, Any]] = {}
    provenance: dict[str, str] = {}
    for variant in VARIANTS:
        run_dir = _complete_run(CONFIG_DIR / f"{variant}.yaml", 400)
        provenance[variant.upper()] = str(run_dir.relative_to(ROOT))
        summary = _load_json(run_dir / "summary/metrics.json")
        row = next(
            item for item in summary["metrics"] if str(item["model_id"]) == model_id
        )
        rows[variant.upper()] = row

    names = {
        "shared-sft-only": "retool-coding-0812-evaluation-shared-sft-only-seed42",
        "c0-step100": "retool-coding-0812-evaluation-c0-seed42-step100",
    }
    train_labels = {
        "shared-sft-only": "shared-sft-only",
        "c0-step100": "c0",
    }
    job_types = {"shared-sft-only": "eval-sft-only", "c0-step100": "eval-c0-step100"}
    descriptions = {
        "shared-sft-only": "Shared neutral SFT-only model evaluated on test prompts C0/C1/C2.",
        "c0-step100": "C0 GRPO step-100 checkpoint evaluated on test prompts C0/C1/C2.",
    }
    model_manifest = _load_json(MODEL_MANIFEST)[model_id]
    train_label = train_labels[model_id]
    log_dir = OUTPUT_ROOT / "swanlab-aggregated" / model_id
    run = swanlab.init(
        project="retool-coding-0812",
        name=names[model_id],
        description=descriptions[model_id],
        config={
            "model": {
                "id": model_id,
                "train_system_prompt": train_label,
                "model_path": model_manifest["model_path"],
                "checkpoint_step": model_manifest["step"],
                "seed": 42,
                "base_model": "Qwen/Qwen3.5-4B",
            },
            "evaluation": {
                "benchmark": "lcb_codegen_retool_0812",
                "tasks_per_prompt": 200,
                "test_prompts": ["C0", "C1", "C2"],
                "temperature": 0.0,
                "max_out_length": 10240,
                "seed": 42,
                "post_hoc": True,
            },
            "provenance": provenance,
            "metric_layout": f"final/train_{train_label}/test_<C0|C1|C2>/<metric>",
        },
        mode="online",
        group="retool-coding-0812-seed42",
        job_type=job_types[model_id],
        tags=[job_types[model_id], "myeval"],
        log_dir=str(log_dir),
    )
    values: dict[str, float] = {}
    for prompt, row in rows.items():
        prefix = f"final/train_{train_label}/test_{prompt}"
        for metric in SWANLAB_REPORT_METRICS:
            value = row.get(metric)
            if isinstance(value, (int, float)):
                values[f"{prefix}/{metric}"] = float(value)
    passes = [float(rows[prompt]["pass_at_1"]) for prompt in ("C0", "C1", "C2")]
    chart = (
        HeatMap()
        .add_xaxis(["C0", "C1", "C2"])
        .add_yaxis(
            "resolve@1",
            [train_label],
            [[index, 0, value] for index, value in enumerate(passes)],
            label_opts=opts.LabelOpts(is_show=True),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="lcb_codegen_retool_0812 Train × Test"),
            visualmap_opts=opts.VisualMapOpts(min_=0, max_=1),
        )
    )
    values["final/resolve_heatmap"] = swanlab.ECharts(chart)
    swanlab.log(values, step=0)
    run_id = run.id
    run.finish(state="success")
    print(f"run_id={run_id}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--variant", choices=VARIANTS)
    run_parser.add_argument("--validate-only", action="store_true")
    run_parser.add_argument("--resume", action="store_true")
    subparsers.add_parser("prepare")
    subparsers.add_parser("finalize")
    upload_parser = subparsers.add_parser("upload-swanlab")
    upload_parser.add_argument("--variant", choices=VARIANTS)
    model_upload_parser = subparsers.add_parser("upload-swanlab-model")
    model_upload_parser.add_argument(
        "--model", choices=("shared-sft-only", "c0-step100"), required=True
    )
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "finalize":
        finalize()
    elif args.command == "upload-swanlab":
        upload_swanlab(args.variant)
    elif args.command == "upload-swanlab-model":
        upload_swanlab_model(args.model)
    else:
        _run(args.variant, validate_only=args.validate_only, resume=args.resume)


if __name__ == "__main__":
    main()
