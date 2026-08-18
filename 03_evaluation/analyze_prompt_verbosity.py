"""Analyze prompt-conditioned verbosity, repetition, and cap-hit behavior.

This script is read-only with respect to experiment artifacts. It loads the
3,600 saved evaluation trajectories, writes a curated Markdown report under
``docs/investigations``, and writes machine-readable CSV/JSON summaries under
``artifacts/analysis``. It performs no network or remote-model calls.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
import re
from statistics import median
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
ANALYSIS_DIR = ARTIFACTS / "analysis"
REPORT_PATH = ROOT / "docs/investigations/prompt-verbosity-and-cap-hit-analysis.md"
CSV_PATH = ANALYSIS_DIR / "prompt-verbosity-cell-metrics.csv"
JSON_PATH = ANALYSIS_DIR / "prompt-verbosity-analysis.json"

MODELS = (
    "base",
    "shared-sft-only",
    "c0",
    "c0-step100",
    "c1",
    "c2",
)
MODEL_LABELS = {
    "base": "Base",
    "shared-sft-only": "SFT-only",
    "c0": "C0@40",
    "c0-step100": "C0@100",
    "c1": "C1@100",
    "c2": "C2@100",
}
PROMPTS = ("C0", "C1", "C2")
SELECTED_MODELS = ("c0", "c1", "c2")
STEP100_MODELS = ("c0-step100", "c1", "c2")
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260814

REVISION_PATTERN = re.compile(
    r"\b(?:but\s+wait|wait|actually|let(?:'|’)s\s+try|try\s+(?:a\s+)?different|"
    r"different\s+approach|doesn(?:'|’)t\s+help|not\s+right|rethink|start\s+over|"
    r"go\s+back)\b",
    flags=re.IGNORECASE,
)
TOKEN_PATTERN = re.compile(
    r"[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|==|!=|<=|>=|//|<<|>>|\*\*|\S"
)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else math.nan


def bootstrap_ci(values: Iterable[float], *, seed_offset: int = 0) -> list[float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return [math.nan, math.nan]
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    draws = rng.choice(array, size=(BOOTSTRAP_SAMPLES, array.size), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def normalize_line(line: str) -> str | None:
    line = line.strip()
    line = re.sub(r"^#+\s*", "", line)
    line = re.sub(r"\s+", " ", line).strip().lower()
    if len(line) < 12:
        return None
    if line.startswith(("```", "<tool_call", "</tool_call", "<function", "</function")):
        return None
    if line.startswith(("<parameter", "</parameter")):
        return None
    return line


def repeated_line_ratio(turn_texts: list[str]) -> float:
    repeated = 0
    total = 0
    for text in turn_texts:
        lines = [normalized for raw in text.splitlines() if (normalized := normalize_line(raw))]
        counts = Counter(lines)
        repeated += sum(count - 1 for count in counts.values() if count > 1)
        total += len(lines)
    return repeated / total if total else 0.0


def repeated_ngram_ratio(turn_texts: list[str], n: int = 8) -> float:
    repeated = 0
    total = 0
    for text in turn_texts:
        tokens = [token.lower() for token in TOKEN_PATTERN.findall(text)]
        ngrams = [tuple(tokens[index : index + n]) for index in range(max(0, len(tokens) - n + 1))]
        counts = Counter(ngrams)
        repeated += sum(count - 1 for count in counts.values() if count > 1)
        total += len(ngrams)
    return repeated / total if total else 0.0


def top_repeated_lines(turn_texts: list[str], limit: int = 6) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for text in turn_texts:
        per_turn = Counter(
            normalized for raw in text.splitlines() if (normalized := normalize_line(raw))
        )
        counts.update(per_turn)
    return [(line, count) for line, count in counts.most_common(limit) if count > 1]


def load_rows() -> list[dict[str, Any]]:
    paths = sorted(ARTIFACTS.glob("evaluation/*/*/artifacts/trajectories/*.json.gz"))
    paths += sorted(ARTIFACTS.glob("evaluation-posthoc/*/*/artifacts/trajectories/*.json.gz"))
    rows: list[dict[str, Any]] = []
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            trajectory = json.load(stream)
        turns = list(trajectory.get("turns") or [])
        turn_texts = [str(turn.get("text") or "") for turn in turns]
        turn_token_counts = [int(turn.get("completion_token_count") or 0) for turn in turns]
        completion_tokens = sum(turn_token_counts)
        format_valid = bool(trajectory.get("final_code") is not None)
        nonfinal_tokens = sum(turn_token_counts[:-1]) if format_valid else completion_tokens
        first_turn_cap = bool(turns and turns[0].get("hit_token_limit"))
        revision_markers = sum(len(REVISION_PATTERN.findall(text)) for text in turn_texts)
        judge = dict(trajectory.get("judge_result") or {})
        relative_path = path.relative_to(ARTIFACTS)
        rows.append(
            {
                "model": str(trajectory["metadata"]["model_id"]).lower(),
                "test_prompt": str(trajectory["metadata"]["system_prompt_id"]).upper(),
                "sample_id": str(trajectory["example"]["instance_id"]),
                "title": str(trajectory["example"].get("question_title") or ""),
                "difficulty": str(trajectory["example"].get("difficulty") or "unknown"),
                "completion_tokens": completion_tokens,
                "first_turn_tokens": turn_token_counts[0] if turn_token_counts else 0,
                "nonfinal_tokens": nonfinal_tokens,
                "final_turn_tokens": turn_token_counts[-1] if format_valid and turn_token_counts else 0,
                "turns": len(turns),
                "line_repeat_ratio": repeated_line_ratio(turn_texts),
                "ngram8_repeat_ratio": repeated_ngram_ratio(turn_texts),
                "revision_markers": revision_markers,
                "revision_markers_per_1k": 1000.0 * revision_markers / max(completion_tokens, 1),
                "cap_hit": float(bool(trajectory.get("hit_token_limit"))),
                "first_turn_cap": float(first_turn_cap),
                "format_valid": float(format_valid),
                "pass_at_1": float(bool(judge.get("resolved"))),
                "duration_seconds": float(trajectory.get("duration_seconds") or 0.0),
                "tool_calls": int(trajectory.get("tool_calls") or 0),
                "valid_tool_calls": int(trajectory.get("valid_tool_calls") or 0),
                "trajectory_path": str(relative_path),
                "turn_texts": turn_texts,
            }
        )
    expected = len(MODELS) * len(PROMPTS) * 200
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} trajectories, found {len(rows)}")
    keys = {(row["model"], row["test_prompt"], row["sample_id"]) for row in rows}
    if len(keys) != expected:
        raise ValueError("Trajectory cells are not uniquely paired")
    if {row["model"] for row in rows} != set(MODELS):
        raise ValueError("Unexpected model set")
    if {row["test_prompt"] for row in rows} != set(PROMPTS):
        raise ValueError("Unexpected test-prompt set")
    return rows


def summarize_cell(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_count = sum(row["format_valid"] for row in rows)
    cap_rows = [row for row in rows if row["cap_hit"] > 0.5]
    total_tokens = sum(row["completion_tokens"] for row in rows)
    final_tokens = sum(row["final_turn_tokens"] for row in rows)
    return {
        "n": len(rows),
        "mean_completion_tokens": mean(row["completion_tokens"] for row in rows),
        "median_completion_tokens": float(median(row["completion_tokens"] for row in rows)),
        "mean_first_turn_tokens": mean(row["first_turn_tokens"] for row in rows),
        "mean_nonfinal_tokens": mean(row["nonfinal_tokens"] for row in rows),
        "final_turn_token_share": final_tokens / total_tokens if total_tokens else math.nan,
        "mean_line_repeat_ratio": mean(row["line_repeat_ratio"] for row in rows),
        "mean_ngram8_repeat_ratio": mean(row["ngram8_repeat_ratio"] for row in rows),
        "mean_revision_markers_per_1k": mean(row["revision_markers_per_1k"] for row in rows),
        "cap_only_line_repeat_ratio": mean(row["line_repeat_ratio"] for row in cap_rows),
        "cap_hit_rate": mean(row["cap_hit"] for row in rows),
        "first_turn_cap_rate": mean(row["first_turn_cap"] for row in rows),
        "format_valid_rate": valid_count / len(rows),
        "pass_at_1": mean(row["pass_at_1"] for row in rows),
        "pass_given_valid": sum(row["pass_at_1"] for row in rows) / max(valid_count, 1.0),
        "mean_duration_seconds": mean(row["duration_seconds"] for row in rows),
        "mean_turns": mean(row["turns"] for row in rows),
        "mean_tool_calls": mean(row["tool_calls"] for row in rows),
    }


def cell_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["test_prompt"])].append(row)
    result = []
    for model in MODELS:
        for prompt in PROMPTS:
            cell = grouped[(model, prompt)]
            if len(cell) != 200:
                raise ValueError(f"Expected 200 rows for {model} x {prompt}, found {len(cell)}")
            result.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "test_prompt": prompt,
                    **summarize_cell(cell),
                }
            )
    return result


def paired_test_prompt_effects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table = {(row["model"], row["sample_id"], row["test_prompt"]): row for row in rows}
    result = []
    metrics = (
        "completion_tokens",
        "nonfinal_tokens",
        "line_repeat_ratio",
        "ngram8_repeat_ratio",
        "revision_markers_per_1k",
        "cap_hit",
        "format_valid",
        "pass_at_1",
    )
    for model_index, model in enumerate(MODELS):
        sample_ids = sorted(row["sample_id"] for row in rows if row["model"] == model and row["test_prompt"] == "C0")
        pairs = [(table[(model, sample_id, "C0")], table[(model, sample_id, "C2")]) for sample_id in sample_ids]
        record: dict[str, Any] = {"model": model, "model_label": MODEL_LABELS[model], "n": len(pairs)}
        for metric_index, metric in enumerate(metrics):
            deltas = [float(right[metric]) - float(left[metric]) for left, right in pairs]
            record[metric] = {
                "delta": mean(deltas),
                "ci95": bootstrap_ci(deltas, seed_offset=model_index * 20 + metric_index),
            }
        record["cap_flips"] = {
            "c0_only": sum(left["cap_hit"] > 0.5 and right["cap_hit"] < 0.5 for left, right in pairs),
            "c2_only": sum(right["cap_hit"] > 0.5 and left["cap_hit"] < 0.5 for left, right in pairs),
            "both": sum(right["cap_hit"] > 0.5 and left["cap_hit"] > 0.5 for left, right in pairs),
            "neither": sum(right["cap_hit"] < 0.5 and left["cap_hit"] < 0.5 for left, right in pairs),
        }
        result.append(record)
    return result


def paired_train_prompt_effects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table = {(row["model"], row["sample_id"], row["test_prompt"]): row for row in rows}
    result = []
    contrasts = (("c0-step100", "c1"), ("c0-step100", "c2"))
    metrics = ("completion_tokens", "nonfinal_tokens", "line_repeat_ratio", "cap_hit", "format_valid", "pass_at_1")
    for prompt_index, prompt in enumerate(PROMPTS):
        sample_ids = sorted(row["sample_id"] for row in rows if row["model"] == "c0-step100" and row["test_prompt"] == prompt)
        for contrast_index, (left_model, right_model) in enumerate(contrasts):
            pairs = [
                (table[(left_model, sample_id, prompt)], table[(right_model, sample_id, prompt)])
                for sample_id in sample_ids
            ]
            record: dict[str, Any] = {
                "contrast": f"{MODEL_LABELS[right_model]}−{MODEL_LABELS[left_model]}",
                "left_model": left_model,
                "right_model": right_model,
                "test_prompt": prompt,
                "n": len(pairs),
            }
            for metric_index, metric in enumerate(metrics):
                deltas = [float(right[metric]) - float(left[metric]) for left, right in pairs]
                record[metric] = {
                    "delta": mean(deltas),
                    "ci95": bootstrap_ci(
                        deltas,
                        seed_offset=200 + prompt_index * 20 + contrast_index * 8 + metric_index,
                    ),
                }
            result.append(record)
    return result


def aggregate_subset(rows: list[dict[str, Any]], models: tuple[str, ...], prompt: str) -> dict[str, float]:
    subset = [row for row in rows if row["model"] in models and row["test_prompt"] == prompt]
    return {
        "completion_tokens": mean(row["completion_tokens"] for row in subset),
        "cap_hit_rate": mean(row["cap_hit"] for row in subset),
        "line_repeat_ratio": mean(row["line_repeat_ratio"] for row in subset),
        "pass_at_1": mean(row["pass_at_1"] for row in subset),
    }


def find_row(rows: list[dict[str, Any]], model: str, sample_id: str, prompt: str) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["model"] == model and row["sample_id"] == sample_id and row["test_prompt"] == prompt
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one row for {model}/{sample_id}/{prompt}, found {len(matches)}")
    return matches[0]


def artifact_link(row: dict[str, Any]) -> str:
    return f"../{row['trajectory_path']}"


def fmt_number(value: float, digits: int = 1) -> str:
    return f"{value:,.{digits}f}"


def fmt_pct(value: float, digits: int = 2) -> str:
    return f"{100.0 * value:.{digits}f}%"


def fmt_delta_ci(effect: dict[str, Any], *, scale: float = 1.0, digits: int = 1, suffix: str = "") -> str:
    delta = scale * float(effect["delta"])
    low, high = (scale * float(value) for value in effect["ci95"])
    return f"{delta:+.{digits}f}{suffix} [{low:+.{digits}f}, {high:+.{digits}f}]"


def write_csv(cells: list[dict[str, Any]]) -> None:
    fieldnames = list(cells[0].keys())
    with CSV_PATH.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cells)


def write_report(
    rows: list[dict[str, Any]],
    cells: list[dict[str, Any]],
    test_effects: list[dict[str, Any]],
    train_effects: list[dict[str, Any]],
) -> None:
    cell_index = {(cell["model"], cell["test_prompt"]): cell for cell in cells}
    selected_c0 = aggregate_subset(rows, SELECTED_MODELS, "C0")
    selected_c2 = aggregate_subset(rows, SELECTED_MODELS, "C2")
    all_c0 = aggregate_subset(rows, MODELS, "C0")
    all_c2 = aggregate_subset(rows, MODELS, "C2")
    first_turn_caps = int(sum(row["first_turn_cap"] for row in rows))
    total_caps = int(sum(row["cap_hit"] for row in rows))

    humidifier_c0 = find_row(rows, "c0-step100", "abc383_a", "C0")
    humidifier_c2 = find_row(rows, "c0-step100", "abc383_a", "C2")
    adjacent_c0 = find_row(rows, "c0-step100", "arc185_e", "C0")
    adjacent_c2 = find_row(rows, "c0-step100", "arc185_e", "C2")
    double_sum = {
        model: find_row(rows, model, "abc384_f", "C0") for model in ("c0", "c1", "c2")
    }
    humidifier_repeats = top_repeated_lines(humidifier_c2["turn_texts"], limit=8)

    lines: list[str] = []
    lines.extend(
        [
            "# ReTool-Coding 0812：Prompt 冗长、重复与 Cap-hit 对比报告",
            "",
            "> 分析对象：最终评测与 post-hoc 最终评测的 3,600 条原始轨迹，seed 42。  ",
            "> 每个 Train × Test cell 含同一批 200 道 LCB-v6 题；生成使用 greedy decoding。  ",
            "> 数据来源：`artifacts/evaluation` 与 `artifacts/evaluation-posthoc` 下逐轨迹 `.json.gz`；统计生成时间：2026-08-14。",
            "",
            "## 0. 结论先行",
            "",
            "这组数据把“prompt 详细导致废话和 cap-hit”拆成两个不同问题：",
            "",
            f"1. **测试时临时换成更详细的 C2 prompt，并没有让入选模型总体变长。** 固定 C0/C1/C2 三个入选权重后，test C0 与 test C2 的平均 completion 分别为 `{selected_c0['completion_tokens']:.1f}` 与 `{selected_c2['completion_tokens']:.1f}` tokens，差值只有 `{selected_c2['completion_tokens'] - selected_c0['completion_tokens']:+.1f}`；cap-hit 反而从 `{fmt_pct(selected_c0['cap_hit_rate'])}` 变为 `{fmt_pct(selected_c2['cap_hit_rate'])}`。",
            f"2. **训练时使用更详细的 prompt，与更长、更易触顶的权重行为明显相关。** 跨三个 test prompt，C0@100 / C1@100 / C2@100 的平均 completion 为 `{mean(cell_index[('c0-step100', p)]['mean_completion_tokens'] for p in PROMPTS):.0f}` / `{mean(cell_index[('c1', p)]['mean_completion_tokens'] for p in PROMPTS):.0f}` / `{mean(cell_index[('c2', p)]['mean_completion_tokens'] for p in PROMPTS):.0f}` tokens，cap-hit 为 `{fmt_pct(mean(cell_index[('c0-step100', p)]['cap_hit_rate'] for p in PROMPTS))}` / `{fmt_pct(mean(cell_index[('c1', p)]['cap_hit_rate'] for p in PROMPTS))}` / `{fmt_pct(mean(cell_index[('c2', p)]['cap_hit_rate'] for p in PROMPTS))}`。其中 C1 的关联最强。",
            f"3. **确实存在由重复或绕路直接导致触顶的原始样例，但不是所有额外 token 都是废话。** 例如 `Humidifier 1` 的同模型配对中，C0 用 865 tokens 通过，C2 在第一轮生成到 10,240 tokens，出现高度重复后被截断；但 `Adjacent GCD` 存在相反方向，C0 触顶而 C2 用 5,561 tokens 通过。",
            f"4. **cap-hit 主要发生在第一次 assistant turn。** `{first_turn_caps}/{total_caps}`（`{fmt_pct(first_turn_caps / total_caps)}`）的 cap-hit 在第一轮发生，通常尚未完成合法工具调用和最终提交。这更像是模型把推理、草稿或循环内容塞进第一次 tool-call payload，而不是看完工具结果后才耗尽预算。",
            "5. **当前证据支持的是行为关联，不是跨 seed 的训练因果定律。** 统一 step100 能排除 checkpoint 步数混杂，但仍只有一个训练 seed；‘废话’指标也只是自动代理，需要结合盲审和强制收束反事实实验。",
            "",
            "## 1. 数据与指标口径",
            "",
            "### 1.1 数据完整性",
            "",
            "- selected final：Base、C0@40、C1@100、C2@100 × C0/C1/C2 test prompt，共 2,400 条。",
            "- post-hoc final：SFT-only、C0@100 × C0/C1/C2 test prompt，共 1,200 条。",
            "- 合计 18 个 cell、3,600 条轨迹；每个 cell 200 条，键为 `model × system_prompt_id × instance_id`，无重复或缺失。",
            "- 只分析 `turns[].text` 与 `turns[].completion_token_count`，不把 system/user prompt 或 tool observation 计入模型冗长与重复。",
            "",
            "### 1.2 三组可测指标",
            "",
            "| 指标组 | 本报告的量化定义 | 解释边界 |",
            "|---|---|---|",
            "| 冗长度 | completion token 均值/中位数、第一轮 tokens、未进入合法最终提交的 tokens、最终提交轮 token 占比 | `未提交 tokens`：有合法 final 时为此前各轮 tokens；无合法 final 时为全部生成 tokens。它衡量未交付生成，不等于全部无用。 |",
            "| 重复与绕路 | 每条轨迹的轮内重复行率、重复 8-gram 率、每千 token 的改口标记次数；另报告 cap 轨迹的重复行率 | 重复只在同一 assistant turn 内计算，避免把工具测试代码复制为 final code 误判为重复。改口标记匹配 `wait/actually/let's try/different approach/not right/rethink` 等短语，是启发式代理。 |",
            "| 最终代价 | cap-hit、第一轮 cap-hit、format-valid、pass@1、合法提交条件通过率、轨迹时延 | pass 和格式来自本地 judge；时延可能受服务负载影响，不作跨批严格因果排序。 |",
            "",
            "重复指标不能单独证明‘废话’：正确程序中的循环结构、公式和必要复核也会重复。最强证据应是‘高重复/多次改口 + 未形成 final + 强制提前收束后正确率恢复’的组合；最后一项需要新增反事实评测。",
            "",
            "## 2. 18-cell 冗长度指标",
            "",
            "| Train model | Test | Mean completion | Median completion | Mean first turn | Mean unsubmitted | Final-turn share |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for cell in cells:
        lines.append(
            f"| {cell['model_label']} | {cell['test_prompt']} | {fmt_number(cell['mean_completion_tokens'], 1)} | "
            f"{fmt_number(cell['median_completion_tokens'], 1)} | {fmt_number(cell['mean_first_turn_tokens'], 1)} | "
            f"{fmt_number(cell['mean_nonfinal_tokens'], 1)} | {fmt_pct(cell['final_turn_token_share'], 1)} |"
        )

    lines.extend(
        [
            "",
            "读表重点：",
            "",
            "- C0@40 在三个 test prompt 下都最短；C1@100 都最长。",
            "- `Mean unsubmitted` 同时受到‘第一轮工具调用较长’与‘触顶后完全没有 final’影响，因此它比总 token 更接近终止失败成本。",
            "- 同一模型横向看 C0/C1/C2 test prompt，没有出现一致的‘越详细越长’阶梯；纵向看训练权重，则 C1/C2，尤其 C1，明显比 C0 更长。",
            "",
            "## 3. 18-cell 重复与绕路代理指标",
            "",
            "| Train model | Test | Repeated lines | Repeated 8-grams | Revision markers / 1K | Cap-only repeated lines |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for cell in cells:
        lines.append(
            f"| {cell['model_label']} | {cell['test_prompt']} | {fmt_pct(cell['mean_line_repeat_ratio'], 1)} | "
            f"{fmt_pct(cell['mean_ngram8_repeat_ratio'], 1)} | {fmt_number(cell['mean_revision_markers_per_1k'], 2)} | "
            f"{fmt_pct(cell['cap_only_line_repeat_ratio'], 1)} |"
        )

    lines.extend(
        [
            "",
            "这些平均值会被少量极端循环显著拉高，因此报告同时保留原始轨迹案例。重复率较低也不代表没有废话：模型可能不断提出不同但均失败的算法，形成长而不重复的 dead-end chain，例如下文的 `Double Sum 2` C2 轨迹。",
            "",
            "## 4. 18-cell 最终代价",
            "",
            "| Train model | Test | Cap-hit | First-turn cap | Format-valid | Pass@1 | Pass given valid | Mean latency |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cell in cells:
        lines.append(
            f"| {cell['model_label']} | {cell['test_prompt']} | {fmt_pct(cell['cap_hit_rate'], 1)} | "
            f"{fmt_pct(cell['first_turn_cap_rate'], 1)} | {fmt_pct(cell['format_valid_rate'], 1)} | "
            f"{fmt_pct(cell['pass_at_1'], 1)} | {fmt_pct(cell['pass_given_valid'], 1)} | "
            f"{fmt_number(cell['mean_duration_seconds'], 1)}s |"
        )

    lines.extend(
        [
            "",
            "最终代价呈现清晰的 trade-off：C0 权重更容易及时交付，但合法代码的条件正确率较低；C1/C2 权重生成更长、触顶更多，但一旦合法提交，正确率明显更高。因此优化目标不能只是缩短输出，而应减少‘未形成 final 的冗长’，保留能提高代码正确性的有效推理。",
            "",
            "## 5. 固定模型与题目：Test C2 − Test C0 配对",
            "",
            "每行使用同一模型的同一 200 道题做配对。token 和重复率报告 C2−C0 的均值与逐题 bootstrap 95% CI；cap flips 中 `C2-only` 表示只有 C2 触顶，`C0-only` 表示只有 C0 触顶。",
            "",
            "| Model | Δ completion tokens | Δ unsubmitted tokens | Δ repeated lines | Δ cap-hit | Cap flips C0-only / C2-only |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for effect in test_effects:
        flips = effect["cap_flips"]
        lines.append(
            f"| {effect['model_label']} | {fmt_delta_ci(effect['completion_tokens'], digits=0)} | "
            f"{fmt_delta_ci(effect['nonfinal_tokens'], digits=0)} | "
            f"{fmt_delta_ci(effect['line_repeat_ratio'], scale=100, digits=1, suffix='pp')} | "
            f"{fmt_delta_ci(effect['cap_hit'], scale=100, digits=1, suffix='pp')} | "
            f"{flips['c0_only']} / {flips['c2_only']} |"
        )

    lines.extend(
        [
            "",
            f"在三个入选模型上聚合，test C2 相对 test C0 只多 `{selected_c2['completion_tokens'] - selected_c0['completion_tokens']:+.1f}` tokens，cap-hit 低 `{100 * (selected_c0['cap_hit_rate'] - selected_c2['cap_hit_rate']):.2f}pp`。在全部六个权重上，C2 平均多 `{all_c2['completion_tokens'] - all_c0['completion_tokens']:+.1f}` tokens，cap-hit 只高 `{100 * (all_c2['cap_hit_rate'] - all_c0['cap_hit_rate']):+.2f}pp`。这些结果不支持 test-time 详细 prompt 是总体 cap 增长的主因。",
            "",
            "同时，各模型都有 C0-only 与 C2-only cap 翻转，说明 prompt 会改变个别题的生成路径；应报告双向案例，而不能只展示支持猜想的样本。",
            "",
            "## 6. 统一 Step 100：Train prompt 对比",
            "",
            "下表固定同一 test prompt，并用同一 200 题比较 C1@100/C2@100 与 C0@100。这样排除了 C0 selected checkpoint 为 step40 的步数混杂。CI 仍只反映题目抽样不确定性，不包含训练 seed 方差。",
            "",
            "| Train contrast | Test | Δ completion tokens | Δ unsubmitted | Δ repeated lines | Δ cap-hit | Δ pass@1 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for effect in train_effects:
        lines.append(
            f"| {effect['contrast']} | {effect['test_prompt']} | "
            f"{fmt_delta_ci(effect['completion_tokens'], digits=0)} | "
            f"{fmt_delta_ci(effect['nonfinal_tokens'], digits=0)} | "
            f"{fmt_delta_ci(effect['line_repeat_ratio'], scale=100, digits=1, suffix='pp')} | "
            f"{fmt_delta_ci(effect['cap_hit'], scale=100, digits=1, suffix='pp')} | "
            f"{fmt_delta_ci(effect['pass_at_1'], scale=100, digits=1, suffix='pp')} |"
        )

    lines.extend(
        [
            "",
            "统一 step 后，C1 相对 C0 的长度与 cap 增量仍最大；C2 的增量更小。这支持‘训练 prompt 塑造了终止/长度行为’，并与 C2 相对 C1 更好的成本折中一致。不过 C1/C2 同时提高 pass，因此额外 token 中既有有效推理，也有终止失败，不能一概删除。",
            "",
            "## 7. 原始轨迹对比",
            "",
            "### 7.1 支持猜想的同模型样例：Humidifier 1",
            "",
            f"同一个 C0@100 权重、同一道题：[`test C0` 原始轨迹]({artifact_link(humidifier_c0)}) 使用 `{humidifier_c0['completion_tokens']}` tokens，合法提交并通过；[`test C2` 原始轨迹]({artifact_link(humidifier_c2)}) 在第一轮达到 `{humidifier_c2['completion_tokens']}` tokens，未形成合法 final。",
            "",
            f"C2 轨迹的轮内重复行率为 `{fmt_pct(humidifier_c2['line_repeat_ratio'], 1)}`。最高频内容包括：",
            "",
        ]
    )
    for repeated_line, count in humidifier_repeats:
        lines.append(f"- `{repeated_line[:120]}`：{count} 次")

    lines.extend(
        [
            "",
            "这条轨迹可以明确归入‘重复循环导致 cap-hit’，而不只是算法本身复杂。",
            "",
            "### 7.2 反方向样例：Adjacent GCD",
            "",
            f"同一个 C0@100 权重、同一道题：[`test C0` 原始轨迹]({artifact_link(adjacent_c0)}) 在第一轮 10,240 tokens 触顶；[`test C2` 原始轨迹]({artifact_link(adjacent_c2)}) 使用 5,561 tokens，形成合法提交并通过。它说明详细 prompt 有时也会把模型从发散路径切换到可完成路径。",
            "",
            "### 7.3 固定 Test C0 的训练分支样例：Double Sum 2",
            "",
            f"在同一道题、同一个 test C0 下，[`C0@40`]({artifact_link(double_sum['c0'])}) 用 `{double_sum['c0']['completion_tokens']}` tokens 提交并通过；[`C1@100`]({artifact_link(double_sum['c1'])}) 和 [`C2@100`]({artifact_link(double_sum['c2'])}) 均在第一轮达到 10,240 tokens 后失败。C1 的重复行率为 `{fmt_pct(double_sum['c1']['line_repeat_ratio'], 1)}`；C2 的重复率较低（`{fmt_pct(double_sum['c2']['line_repeat_ratio'], 1)}`），但连续尝试多个未完成方向。这展示了两种不同的‘废话’：显式循环，以及不重复但持续换路的 dead-end chain。",
            "",
            "该案例仍不是训练 prompt 的独立统计证据；总体判断应以前一节的 200 题配对和多 seed 复验为准。",
            "",
            "## 8. 如何把猜想升级为更强的验证",
            "",
            "1. **盲审分层样本。** 从 C2-only cap、C0-only cap、both-cap、neither-cap 各抽取相同数量轨迹，隐藏模型和 prompt，标注 problem restatement、重复推导、算法换路、工具草稿过长、已有可用代码但未提交等类别；报告双标注一致率。",
            "2. **强制收束反事实。** 固定现有 checkpoint 和题目，在 4K/6K/8K tokens 或检测到高重复时注入‘立即提交当前最佳代码’，与原轨迹做同题配对。若 cap 降低且 pass 不降，才说明被删掉的 token 主要是无用冗长。",
            "3. **训练消融。** 固定 C2 策略，比较当前 reward、termination cue、termination-aware shaping 及二者组合；先做 20–40 step canary。主指标应同时包含 pass@1、cap-hit、未提交 tokens 和合法提交条件通过率。",
            "4. **多 seed。** 当前训练 prompt 对比只有 seed 42。至少补 seed 43/44，并在同一 seed 内让各分支共享初始化、训练题顺序和采样随机流。",
            "",
            "## 9. 复现文件",
            "",
            f"- 分析脚本：[`{Path(__file__).name}`]({Path(__file__).name})",
            f"- 18-cell CSV：[`{CSV_PATH.name}`]({CSV_PATH.name})",
            f"- 完整机器可读结果：[`{JSON_PATH.name}`]({JSON_PATH.name})",
            "- 原始轨迹：`../evaluation/*/*/artifacts/trajectories/*.json.gz` 与 `../evaluation-posthoc/*/*/artifacts/trajectories/*.json.gz`。",
            "",
            "统计使用每道题配对的 10,000 次 bootstrap，seed `20260814`。自动重复指标按 assistant turn 分段，tool observation 不进入统计。",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = load_rows()
    cells = cell_summaries(rows)
    test_effects = paired_test_prompt_effects(rows)
    train_effects = paired_train_prompt_effects(rows)
    write_csv(cells)
    payload = {
        "metadata": {
            "n_trajectories": len(rows),
            "n_cells": len(cells),
            "tasks_per_cell": 200,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "assistant_only_text_metrics": True,
            "repeat_scope": "within_each_assistant_turn",
        },
        "cells": cells,
        "paired_test_prompt_c2_minus_c0": test_effects,
        "paired_train_prompt_step100": train_effects,
    }
    JSON_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(rows, cells, test_effects, train_effects)
    print(f"wrote {REPORT_PATH}")
    print(f"wrote {CSV_PATH}")
    print(f"wrote {JSON_PATH}")


if __name__ == "__main__":
    main()
