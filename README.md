# retool-coding-0812

这是 ReTool Coding 的 seed 42 独立复现实验：先用 shared-neutral SFT 建立共同初态，再分别用 C0/C1/C2 system prompt 进行 GRPO，最后做跨 prompt checkpoint 选择和 12-cell 评测。冻结方案为 `e3 + L10K/20K + P1`。

第一次阅读建议按以下顺序：

1. 本页：环境、目录和运行入口。
2. [`docs/code-navigation-guide.md`](docs/code-navigation-guide.md)：重要程序的职责与数据流。
3. [`docs/final-experiment-summary.md`](docs/final-experiment-summary.md)：实验方法、结果和结论。
4. [`docs/README.md`](docs/README.md)：全部报告与结果文档索引。

## 交接说明

Git 仓库只包含源码、冻结配置、测试和精选报告。`artifacts/` 中约 1.4 GB 的 checkpoint、轨迹、SQLite、日志和机器可读分析结果是本地生成物，不会上传；本地文件仍然保留。

完整复现还需要同级的 `07-retool-lcb-mini` 项目，因为数据集和 shared-SFT 源轨迹没有重复复制进本仓库。准备方式与所需路径见 [`inputs/README.md`](inputs/README.md)。此外，`pyproject.toml` 把 `creator-myeval` 指向工作区内的 `../../../myeval`，队友需要使用相同工作区布局或调整为自己的 MyEval 安装位置。

本目录只读取以下冻结输入：

- `inputs/formal-v6`：指向原 400/60/100/200 数据的只读符号链接，避免复制约 2.3 GB。
- `inputs/calibration-gate.json`：已经通过的 calibration/smoke 证据。
- `inputs/protocol-selected.yaml`：冻结协议及其指纹证据。
- `../07-retool-lcb-mini/artifacts/training/shared-cold-start/source/`：300 条严格通过的 shared-neutral SFT 源轨迹。

## 目录

```text
01_train/             # e3 shared-neutral SFT，以及 C0/C1/C2 GRPO
02_checkpoint_dev/    # 五个 checkpoint 的完整跨 prompt dev 评测与选择
03_evaluation/        # Base + C0/C1/C2 的最终 12-cell 评测与汇总
configs/experiment.yaml
                      # 唯一人工维护的冻结配置真源
src/retool_coding_0812/
                      # prompt、rollout、判题、Docker、指标和配置生成共享实现
inputs/               # 冻结输入清单、校准证据及外部数据说明
docs/                 # 交接文档、实验报告、专项调查和结果矩阵
tests/                # checkpoint、恢复、并发与汇总逻辑测试
artifacts/            # 本地生成物；Git 中仅保留 .gitkeep
```

## 环境与记录

- Python 3.11–3.13；锁定 `pytrio==0.2.5`、`swanlab==0.9.2`。
- 推荐使用项目现有 Python 3.12 环境；命令统一使用 `.venv/bin/python`，并设置 `PYTHONPATH=src`。
- Docker CLI 通过 `PATH` 查找；正式 gate 还会核对镜像 ID 与隔离能力记录。
- SwanLab 固定为 online，project `retool-coding-0812`，group `retool-coding-0812-seed42`。
- 所有脚本都从 `configs/experiment.yaml` 构建参数，不提供 seed、步数、长度或学习率覆盖入口。

## 执行顺序

```bash
cd /path/to/retool-coding-0812
export PYTHONPATH=src

# 1. 本地冻结配置与校准 gate 验证；不会联系 PyTRIO。
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python 01_train/run_all.py --gate-only

# 2. 重新训练 seed-42 e3 shared SFT，再并行训练 C0/C1/C2。
# 首次运行与中断后的重启都推荐使用 --resume；没有 checkpoint 时会从当前阶段头部安全开始。
.venv/bin/python 01_train/run_all.py --resume

# 3. 收集 step 20/40/60/80/100，生成并运行 checkpoint-dev。
.venv/bin/python 02_checkpoint_dev/build_model_manifest.py \
  --checkpoint-dir artifacts/training/checkpoints \
  --output artifacts/training/checkpoints/models.json
.venv/bin/python 02_checkpoint_dev/make_configs.py
.venv/bin/python 02_checkpoint_dev/run.py --validate-only
.venv/bin/python 02_checkpoint_dev/run.py

# 4. 用三个实际 MyEval run 目录选择 checkpoint。
.venv/bin/python 02_checkpoint_dev/select_checkpoints.py \
  --myeval-runs <C0_RUN_DIR> <C1_RUN_DIR> <C2_RUN_DIR> \
  --output artifacts/checkpoint_dev/selected-models.json

# 5. 生成并运行最终 12-cell × 200 题评测。
.venv/bin/python 03_evaluation/make_config.py
.venv/bin/python 03_evaluation/run.py --validate-only
.venv/bin/python 03_evaluation/run.py
.venv/bin/python 03_evaluation/summarize.py \
  --myeval-runs <C0_PROMPT_RUN_DIR> <C1_PROMPT_RUN_DIR> <C2_PROMPT_RUN_DIR> \
  --output artifacts/analysis/final-matrix.md
```

### L16K/24K post-hoc length ablation

The inference-only length ablation keeps the frozen checkpoints, 200-task test
set, seed 42, greedy decoding, P1 tool protocol, and Docker/judge limits.  It
raises the per-assistant cap to 16,384 tokens and the full trajectory cap to
24,576 tokens.  The MyEval orchestration timeout is 600 seconds so that a valid
16K generation is not cancelled by the original 300-second task watchdog.

```bash
export PYTHONPATH=src
.venv/bin/python 03_evaluation/make_config.py \
  --experiment-tag l16k-t24k \
  --max-assistant-tokens 16384 \
  --max-trajectory-tokens 24576 \
  --timeout-seconds 600
.venv/bin/python 03_evaluation/run.py \
  --config-dir configs/generated/evaluation-l16k-t24k \
  --validate-only
.venv/bin/python 03_evaluation/run.py \
  --config-dir configs/generated/evaluation-l16k-t24k
```

Results are isolated under `artifacts/evaluation-l16k-t24k/`.  After all three
prompt shards complete, run `03_evaluation/summarize.py`,
`03_evaluation/audit_length_ablation.py`, and
`03_evaluation/analyze_length_ablation.py`.  The audit validates every raw
trajectory's messages, token IDs, logprobs, tool observations, final code,
judge result, and MyEval provenance before the comparison is accepted.

训练在 step 20 检查累计 nondegenerate group rate 和 skipped-update rate；未通过会立即终止。shared SFT 和 GRPO 都每 20 个 optimizer step 保存包含 optimizer 的完整 state。恢复时以最近完整 checkpoint 为权威，checkpoint 后尚未固化的本地 metrics、轨迹或 checkpoint 记录会移入对应 run 的 `recovery/` 目录，再确定性重跑。

整条训练链自动跳过已经完成的 shared SFT/C0/C1/C2，并恢复未完成阶段：

```bash
.venv/bin/python 01_train/run_all.py --resume
```

也可以只恢复某个阶段：

```bash
.venv/bin/python 01_train/shared_sft.py --resume
.venv/bin/python 01_train/run.py --variant c0 --resume

# 或显式指定 checkpoint：
.venv/bin/python 01_train/run.py --variant c0 \
  --resume-checkpoint artifacts/training/checkpoints/retool-coding-0812-c0-seed42-step-20.json
```

checkpoint-dev 和最终评测由 MyEval 的 SQLite 保存逐任务状态。首次运行不加 `--resume`；中断后在相同配置和代码指纹下使用 `--resume`，只执行尚未完成的任务。配置或实现改变时会拒绝错误复用：

```bash
.venv/bin/python 02_checkpoint_dev/run.py --resume
.venv/bin/python 03_evaluation/run.py --resume
```

`01_train/run_all.py` 在 shared SFT 完成后并行启动 C0/C1/C2；checkpoint-dev
同样并行启动三个训练分支的评测。最终 evaluation 按测试 prompt 分为 C0/C1/C2
三个并行 MyEval run，汇总器会合并成完整的 12-cell 配对统计。任一并行子进程失败时，
编排器会终止其余子进程；重新运行 `--resume` 会分别从各自 checkpoint 或 SQLite
状态继续。

## `01_train` 监督器

监督器只覆盖 shared SFT 和 C0/C1/C2 GRPO，不会启动 checkpoint-dev 或 evaluation。状态、日志与锁均写入 `artifacts/training/`：

```bash
# 初始化并核对已有训练产物
.venv/bin/python 01_train/supervisor.py initialize

# 单实例后台启动 run_all.py --resume
.venv/bin/python 01_train/supervisor.py launch

# heartbeat 使用：读取增量状态、更新关键报告，并仅恢复可重试故障
.venv/bin/python 01_train/supervisor.py check --recover --update-report
```

结构化权威状态是 `artifacts/training/supervisor-status.json`。普通检查不会读取整份实验报告或所有轨迹；关键 checkpoint、gate、阶段切换、异常和完成事件才会更新 [`docs/experiment-record-and-results.md`](docs/experiment-record-and-results.md)。
