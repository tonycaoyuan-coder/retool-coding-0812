# ReTool Coding 0812 实验记录

> 本文档是 `retool-coding-0812` 的配置与结果持续记录。当前只运行 seed 42。所有结果必须引用本目录下的本地权威 artifact 与 SwanLab run ID；实验尚未完成的字段保持“待运行”，不得用旧实验结果填充。

## 1. 实验目标与状态

- 目标：复现 `07-retool` 冻结的正式训练方案，完成 seed 42 的 C0/C1/C2 GRPO、checkpoint-dev 选择和最终跨 prompt 评测。
- calibration：不重跑；复用已通过的 `e3 + L10K/20K + P1` hard-gate 证据。
- train：已完成 shared SFT 与 C0/C1/C2 seed-42 GRPO，三个分支均到 step 100。
- checkpoint-dev：已完成；C0/C1/C2 每支 1,500/1,500 completed，已生成 `artifacts/checkpoint_dev/selected-models.json`。
- evaluation：已完成；C0/C1/C2 三个 prompt shard 均为 800/800，最终矩阵已生成。
- 配置真源：`configs/experiment.yaml`。
- 冻结日期：2026-08-12（Asia/Shanghai）。

## 2. 冻结输入

| 项目 | 正式取值 |
|---|---|
| 数据集 | `livecodebench/code_generation_lite` / `v6` |
| revision | `a16d03780493b939b3601fb9da2ac3ed2b23caa2` |
| 数据划分 | 400 train / 60 calibration / 100 checkpoint-dev / 200 test |
| 数据 manifest SHA-256 | `b58f1bab9c78052e72c13f1a9d0d1f476fff1099c09cae6e1b1e9423835d14d1` |
| shared-neutral SFT 源 | 300 个不同题目的严格全测试通过、格式合法、未截断轨迹 |
| SFT 源 manifest SHA-256 | `245769e9ee9826772dc82fb44835f3e723356370e9403792fe0e31360cfb4c5d` |
| calibration 选择 | `e3 + L10K/20K + P1` |
| calibration experiment fingerprint | `f9f5159a48c294affeda51bd9e276d0d4682f44f7e95a47ad08fef9d5b105a25` |
| calibration gate | C0/C1/C2 全部通过 |

## 3. 共享初始化：e3 shared-neutral SFT

shared SFT 属于冻结正式训练初始化，不属于本次跳过的 calibration 搜索；因此在新实验中用同一份冻结 300 条源轨迹重新训练一次 seed 42 初态。

| 参数 | 取值 |
|---|---:|
| Base | `Qwen/Qwen3.5-4B` |
| seed | 42 |
| LoRA rank | 32 |
| epochs | 3 |
| datums | 300 |
| optimizer steps | 114（按冻结 micro-batch packing 预期） |
| loss | cross-entropy |
| mask | assistant-only，手工 autoregressive shift |
| learning rate | `1e-5` |
| Adam beta1 / beta2 | `0.9 / 0.95` |
| max context | 12,288 |
| micro-batch items | 最多 8 |
| padded-token 上限 | 65,536 |
| recipe fingerprint | `6bf17aed47f7be99b58c19ede75ab016b73c17213768d60322642e3a224b0e1d` |

## 4. 正式 GRPO 配置

| 类别 | 正式取值 |
|---|---|
| 分支 | C0 / C1 / C2；只改变 system prompt |
| seed | 42 only |
| 每分支训练 | 100 steps |
| 每步规模 | 4 questions × group 8 = 32 trajectories |
| 每分支总轨迹 | 3,200；三分支共 9,600 |
| 题目使用 | 400 train 题固定顺序，每分支恰好使用一次 |
| rollout sampling | temperature `1.0`，top_p `1.0` |
| 协议 | P1：最多 1 次 `run_python` / 2 assistant turns |
| 长度 | assistant 10,240；trajectory 20,480；tool response 512 |
| reward | 严格格式非法 `-0.1`；格式合法时为通过测试用例比例；全通过为 1.0 |
| advantage | 同题 8 条轨迹的 `reward - group_mean_reward` |
| loss | PyTRIO `importance_sampling` |
| optimizer | Adam，LR `4e-5`，beta1/beta2 `0.9/0.95` |
| micro-batch | 最多 16 datums；padded tokens ≤65,536 |
| checkpoints | step 20 / 40 / 60 / 80 / 100 |
| step-20 early gate | 累计 nondegenerate ≥50%；skipped update ≤25% |
| judge sandbox | Python 3.11；断网；只读根；1 CPU；512 MiB；64 pids |
| timeout | tool 5s；case 3s；whole judge 30s |

## 5. Checkpoint-dev 配置

- 数据：100 checkpoint-dev 题。
- 候选：每个分支的 step 20/40/60/80/100。
- 每个候选在 C0/C1/C2 三种 prompt 下完整评测，不只看匹配 prompt。
- 生成：greedy，temperature 0.0，top_p 1.0，top_k -1，n=1，seed 42。
- 协议/长度/沙箱：与正式冻结 P1 和 L10K/20K 相同。
- 每分支轨迹：5 checkpoints × 3 prompts × 100 = 1,500；三分支共 4,500。
- 选择顺序：跨 prompt average pass@1 → worst pass@1 → average case-pass → 更早 checkpoint。

## 6. Final evaluation 配置

- 数据：200 道 temporally held-out test 题。
- 模型：Base + checkpoint-dev 选出的 C0/C1/C2。
- prompt：每个模型分别测试 C0/C1/C2，共 12 cells、2,400 条轨迹。
- 生成：greedy，temperature 0.0，top_p 1.0，top_k -1，seed 42，n=1，stop=[]。
- 执行并发：4；单任务超时 300s；最多重试 2 次；退避 1s。
- 统计：10,000 次 paired bootstrap、95% CI、exact McNemar；同时报告 Average、Worst、OverfitGap。

## 7. SwanLab

| 字段 | 取值 |
|---|---|
| enabled | true |
| mode | online |
| project | `retool-coding-0812` |
| group | `retool-coding-0812-seed42` |
| jobs | `shared-sft`、`train-C0/C1/C2`、`checkpoint-dev-c0/c1/c2`、`evaluation` |

## 8. 运行结果

### Shared SFT

- 状态：运行中（监督器单实例启动，后续从每 20 步完整 state 恢复）
- SwanLab run ID：`chjlx0h7`
- state path：待记录
- sampler weights：待记录
- 实际 optimizer steps / duration：待记录

### GRPO

| 分支 | 状态 | SwanLab run ID | step-20 gate | 选定/最终 checkpoint | 备注 |
|---|---|---|---|---|---|
| C0 | 已完成（100/100 steps） | `a71cec9q` | 通过：nondegenerate `77.5%`，skipped update `0.0%` | step 40 | checkpoint-dev 的 Average pass@1 与 step 80 同为 `0.5667`，以更高 Worst pass@1（`0.5600`）胜出 |
| C1 | 已完成（100/100 steps） | `eyqu4mod` | 通过：nondegenerate `77.5%`，skipped update `0.0%` | step 100 | checkpoint-dev 的 Average pass@1 为本分支最高（`0.6200`） |
| C2 | 已完成（100/100 steps） | `2hggeal6` | 通过：nondegenerate `82.5%`，skipped update `0.0%` | step 100 | checkpoint-dev 的 Average pass@1 为本分支最高（`0.6433`） |

### Checkpoint-dev

每个 checkpoint 在同一组 100 道 dev 题上分别使用 C0/C1/C2 prompt 评测，因此每行包含 300 条配对轨迹。`Average pass@1` 是三个 prompt 的算术平均，`Worst pass@1` 是其中最低值，`Average case-pass` 是三个 prompt 的测试用例通过率均值。选择按 `(Average pass@1, Worst pass@1, Average case-pass, 更早 step)` 依次进行字典序比较（因为我们优先希望模型有能力考虑各种边界情况、完整解答问题，而不只是通过更多测试用例）；加粗行是最终选择。

| 训练分支 | Step | C0 prompt pass@1 | C1 prompt pass@1 | C2 prompt pass@1 | Average pass@1 | Worst pass@1 | Average case-pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 20 | 0.5300 | 0.5300 | 0.5900 | 0.5500 | 0.5300 | 0.6241 |
| **C0** | **40** | **0.5700** | **0.5700** | **0.5600** | **0.5667** | **0.5600** | **0.6405** |
| C0 | 60 | 0.4800 | 0.5200 | 0.5500 | 0.5167 | 0.4800 | 0.6146 |
| C0 | 80 | 0.6000 | 0.5500 | 0.5500 | 0.5667 | 0.5500 | 0.6595 |
| C0 | 100 | 0.5600 | 0.5800 | 0.5400 | 0.5600 | 0.5400 | 0.6378 |
| C1 | 20 | 0.4800 | 0.4900 | 0.4800 | 0.4833 | 0.4800 | 0.5761 |
| C1 | 40 | 0.5700 | 0.5900 | 0.5300 | 0.5633 | 0.5300 | 0.6519 |
| C1 | 60 | 0.6300 | 0.6100 | 0.6000 | 0.6133 | 0.6000 | 0.6775 |
| C1 | 80 | 0.5900 | 0.6100 | 0.6000 | 0.6000 | 0.5900 | 0.6650 |
| **C1** | **100** | **0.5900** | **0.6400** | **0.6300** | **0.6200** | **0.5900** | **0.6840** |
| C2 | 20 | 0.5300 | 0.5100 | 0.5500 | 0.5300 | 0.5100 | 0.6089 |
| C2 | 40 | 0.4700 | 0.5000 | 0.5200 | 0.4967 | 0.4700 | 0.5661 |
| C2 | 60 | 0.5200 | 0.5700 | 0.5700 | 0.5533 | 0.5200 | 0.6199 |
| C2 | 80 | 0.5300 | 0.5900 | 0.5600 | 0.5600 | 0.5300 | 0.6295 |
| **C2** | **100** | **0.6300** | **0.6700** | **0.6300** | **0.6433** | **0.6300** | **0.7176** |

选择依据：C0 的 step 40 与 step 80 都是 `170/300 = 0.5667`，按第二关键字 Worst pass@1，step 40 的 `0.5600` 高于 step 80 的 `0.5500`，因此选择 step 40；C1 和 C2 的 step 100 均拥有各自分支最高的 Average pass@1，直接入选。

| 分支 | 选中 step | Average pass@1 | Worst pass@1 | Case-pass | MyEval run | SwanLab run ID |
|---|---:|---:|---:|---:|---|---|
| C0 | 40 | 0.5667 | 0.5600 | 0.6405 | `20260812T153529Z-retool-coding-0812-checkpoint-dev-c0-seed42-8e5a4498` | `ckk2zdj0` |
| C1 | 100 | 0.6200 | 0.5900 | 0.6840 | `20260812T153529Z-retool-coding-0812-checkpoint-dev-c1-seed42-59aceb20` | `6fuljxi4` |
| C2 | 100 | 0.6433 | 0.6300 | 0.7176 | `20260812T153529Z-retool-coding-0812-checkpoint-dev-c2-seed42-526f0e53` | `gaplx8d5` |

### Final matrix

| Train \\ Test | C0 | C1 | C2 | Average | Worst | OverfitGap |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 0.360 | 0.340 | 0.315 | 0.338 | 0.315 | 0.032 |
| C1 | 0.385 | 0.385 | 0.375 | 0.382 | 0.375 | 0.005 |
| C2 | 0.405 | 0.360 | 0.375 | 0.380 | 0.360 | -0.008 |
| Base | 0.250 | 0.240 | 0.255 | 0.248 | 0.240 | — |

最终评测已完成：三路 MyEval run 各 `800/800` completed，均为 `pending=0`、`running=0`、`failed=0`；共 `2,400/2,400`。完整 paired bootstrap CI 与 exact McNemar 结果见 [`docs/results/final-evaluation-matrix.md`](results/final-evaluation-matrix.md)。C1 的平均分最高（`0.382`），C2 在 C0 prompt 的单格分最高（`0.405`）。

## 9. 运行日志

### 2026-08-12 — 目录与冻结配置建立

- 新建与 `07-retool-lcb-mini` 同级的 `retool-coding-0812`。
- 把 train、checkpoint-dev、evaluation 拆分为独立目录；删除 calibration 搜索、数据下载、旧 pilot 报告与历史恢复编排代码。
- 固定 seed 42，并把 SwanLab 切换为 online。
- 本地验证：3 个单元测试通过；C0/C1/C2 本地 hard gate 全部通过，指纹均为 `f9f5159a48c294affeda51bd9e276d0d4682f44f7e95a47ad08fef9d5b105a25`。
- 环境预检：Python 3.12.12、`pytrio==0.2.5`、`swanlab==0.9.2`；SwanLab API 连通且用户 `tonycaoyuan` 登录有效；PyTRIO 默认环境显示已登录。
- 待完成：实际 shared SFT、GRPO、checkpoint-dev 与 evaluation。

### 2026-08-12 — 远程训练被计费状态阻塞

- 执行：`PYTHONPATH=src .venv/bin/python -u 01_train/shared_sft.py`。
- 结果：PyTRIO 在 TrainingRun 创建前返回 HTTP 409 `billing_insufficient_balance`，`retryable=false`。
- service session：`session_w48kv8c5uvue`。
- 权威边界：0 training steps、0 checkpoints、无 SwanLab run、无残留 trainer；未启动 C0/C1/C2 GRPO 或任何评测。
- 决策：不循环重试非重试型计费错误。恢复 workspace 余额后，执行 `PYTHONPATH=src .venv/bin/python 01_train/run_all.py --resume`；本次没有 checkpoint，会从 shared SFT 第一步安全启动，后续中断则自动使用最近完整 checkpoint。
- blocker artifact：`artifacts/training/blocker-20260812.json`。

### 2026-08-12 — 补齐全流程断点续传

- shared SFT 与 C0/C1/C2 GRPO 均支持 `--resume` 自动发现最近完整 state，并通过 `create_training_client_from_state_with_optimizer()` 恢复模型与 optimizer。
- 每 20 步 checkpoint 后未固化的 metrics/轨迹会先移入 run 内的 `recovery/` 目录，再从 checkpoint 确定性重跑；不会静默覆盖诊断证据。
- `run_all.py --resume` 会校验并跳过完成阶段、恢复未完成阶段；checkpoint-dev 和 evaluation 包装入口向 MyEval 转发 `--resume`。
- 验证：7 个单元测试通过；C0/C1/C2 Docker hard gate 全部通过，冻结指纹仍为 `f9f5159a48c294affeda51bd9e276d0d4682f44f7e95a47ad08fef9d5b105a25`。

### 2026-08-12 — 启动 `01_train` 正式监督训练

- PyTRIO 余额已由用户确认恢复；启动前 Docker 镜像、10 个单元测试与 C0/C1/C2 hard gate 全部通过。
- 使用 `PYTHONPATH=src .venv/bin/python 01_train/supervisor.py launch` 单实例启动 `run_all.py --resume`；监督状态写入 `artifacts/training/supervisor-status.json`。
- Shared SFT 已开始训练，SwanLab run ID 为 `chjlx0h7`；首次确认时完成 step 4/114。
- 旧监督自动化已切换到当前 `retool-coding-0812/01_train`，使用 GPT-5.6 Terra、medium reasoning，每 45 分钟读取增量结构化状态。监督止于 C0/C1/C2 step 100，不启动 checkpoint-dev 或 evaluation。

### 2026-08-12T04:49:26.782483+00:00 — 01_train supervisor event

- `{"kind": "milestones", "milestones": [{"id": "shared-sft:checkpoint:20", "kind": "checkpoint", "path": "/Users/caoyuan/MyWork/Projects/Creator+/retool/retool-coding/retool-coding-0812/artifacts/training/shared-sft/seed42/checkpoints/step-20.json", "stage": "shared-sft", "step": 20}, {"id": "shared-sft:checkpoint:40", "kind": "checkpoint", "path": "/Users/caoyuan/MyWork/Projects/Creator+/retool/retool-coding/retool-coding-0812/artifacts/training/shared-sft/seed42/checkpoints/step-40.json", "stage": "shared-sft", "step": 40}], "observed_at": "2026-08-12T04:49:26.782353+00:00", "stage": "shared-sft"}`

### 2026-08-12T05:32:33.285876+00:00 — 01_train supervisor event

- `{"kind": "milestones", "milestones": [{"id": "shared-sft:checkpoint:100", "kind": "checkpoint", "path": "/Users/caoyuan/MyWork/Projects/Creator+/retool/retool-coding/retool-coding-0812/artifacts/training/shared-sft/seed42/checkpoints/step-100.json", "stage": "shared-sft", "step": 100}, {"id": "shared-sft:checkpoint:60", "kind": "checkpoint", "path": "/Users/caoyuan/MyWork/Projects/Creator+/retool/retool-coding/retool-coding-0812/artifacts/training/shared-sft/seed42/checkpoints/step-60.json", "stage": "shared-sft", "step": 60}, {"id": "shared-sft:checkpoint:80", "kind": "checkpoint", "path": "/Users/caoyuan/MyWork/Projects/Creator+/retool/retool-coding/retool-coding-0812/artifacts/training/shared-sft/seed42/checkpoints/step-80.json", "stage": "shared-sft", "step": 80}, {"id": "shared-sft:complete", "kind": "stage_complete", "stage": "shared-sft", "step": 114, "swanlab_run_id": "chjlx0h7"}], "observed_at": "2026-08-12T05:32:33.285472+00:00", "stage": "c0"}`

### 2026-08-12T06:17:18.917103+00:00 — 01_train supervisor event

- `{"kind": "milestones", "milestones": [{"gate": {"max_skipped_update_rate": 0.25, "min_nondegenerate_group_rate": 0.5, "nondegenerate_group_rate": 0.775, "passed": true, "skipped_update_rate": 0.0, "steps": 20}, "id": "c0:early-gate", "kind": "early_gate", "stage": "c0", "step": 20}, {"id": "c0:checkpoint:20", "kind": "checkpoint", "path": "/Users/caoyuan/MyWork/Projects/Creator+/retool/retool-coding/retool-coding-0812/artifacts/training/checkpoints/retool-coding-0812-c0-seed42-step-20.json", "stage": "c0", "step": 20}], "observed_at": "2026-08-12T06:17:18.916858+00:00", "stage": "c0"}`

### 2026-08-12T07:13:16.527676+00:00 — 01_train supervisor event

- `{"kind": "milestones", "milestones": [{"id": "c0:checkpoint:40", "kind": "checkpoint", "path": "/Users/caoyuan/MyWork/Projects/Creator+/retool/retool-coding/retool-coding-0812/artifacts/training/checkpoints/retool-coding-0812-c0-seed42-step-40.json", "stage": "c0", "step": 40}], "observed_at": "2026-08-12T07:13:16.527502+00:00", "stage": "grpo-parallel"}`

### 2026-08-12T08:45:25.362887+00:00 — 01_train supervisor event

- `{"kind": "milestones", "milestones": [{"id": "c0:checkpoint:60", "kind": "checkpoint", "path": "/Users/caoyuan/MyWork/Projects/Creator+/retool/retool-coding/retool-coding-0812/artifacts/training/checkpoints/retool-coding-0812-c0-seed42-step-60.json", "stage": "c0", "step": 60}, {"gate": {"max_skipped_update_rate": 0.25, "min_nondegenerate_group_rate": 0.5, "nondegenerate_group_rate": 0.775, "passed": true, "skipped_update_rate": 0.0, "steps": 20}, "id": "c1:early-gate", "kind": "early_gate", "stage": "c1", "step": 20}, {"id": "c1:checkpoint:20", "kind": "checkpoint", "path": "/Users/caoyuan/MyWork/Projects/Creator+/retool/retool-coding/retool-coding-0812/artifacts/training/checkpoints/retool-coding-0812-c1-seed42-step-20.json", "stage": "c1", "step": 20}, {"gate": {"max_skipped_update_rate": 0.25, "min_nondegenerate_group_rate": 0.5, "nondegenerate_group_rate": 0.825, "passed": true, "skipped_update_rate": 0.0, "steps": 20}, "id": "c2:early-gate", "kind": "early_gate", "stage": "c2", "step": 20}, {"id": "c2:checkpoint:20", "kind": "checkpoint", "path": "/Users/caoyuan/MyWork/Projects/Creator+/retool/retool-coding/retool-coding-0812/artifacts/training/checkpoints/retool-coding-0812-c2-seed42-step-20.json", "stage": "c2", "step": 20}], "observed_at": "2026-08-12T08:45:25.362532+00:00", "stage": "grpo-parallel"}`

### 2026-08-12T10:15:38.307500+00:00 — 01_train supervisor event

- `{"kind": "milestones", "milestones": [{"id": "c0:checkpoint:80", "kind": "checkpoint", "path": "/Users/caoyuan/MyWork/Projects/Creator+/retool/retool-coding/retool-coding-0812/artifacts/training/checkpoints/retool-coding-0812-c0-seed42-step-80.json", "stage": "c0", "step": 80}, {"id": "c1:checkpoint:40", "kind": "checkpoint", "path": "/Users/caoyuan/MyWork/Projects/Creator+/retool/retool-coding/retool-coding-0812/artifacts/training/checkpoints/retool-coding-0812-c1-seed42-step-40.json", "stage": "c1", "step": 40}, {"id": "c2:checkpoint:40", "kind": "checkpoint", "path": "/Users/caoyuan/MyWork/Projects/Creator+/retool/retool-coding/retool-coding-0812/artifacts/training/checkpoints/retool-coding-0812-c2-seed42-step-40.json", "stage": "c2", "step": 40}], "observed_at": "2026-08-12T10:15:38.307238+00:00", "stage": "grpo-parallel"}`

### 2026-08-12T11:00:11.527382+00:00 — 01_train supervisor event

- `{"kind": "milestones", "milestones": [{"id": "c0:checkpoint:100", "kind": "checkpoint", "path": "/Users/caoyuan/MyWork/Projects/Creator+/retool/retool-coding/retool-coding-0812/artifacts/training/checkpoints/retool-coding-0812-c0-seed42-step-100.json", "stage": "c0", "step": 100}, {"id": "c1:checkpoint:60", "kind": "checkpoint", "path": "/Users/caoyuan/MyWork/Projects/Creator+/retool/retool-coding/retool-coding-0812/artifacts/training/checkpoints/retool-coding-0812-c1-seed42-step-60.json", "stage": "c1", "step": 60}], "observed_at": "2026-08-12T11:00:11.527022+00:00", "stage": "grpo-parallel"}`

### 2026-08-12T11:45:24.319211+00:00 — 01_train supervisor event

- `{"kind": "milestones", "milestones": [{"id": "c2:checkpoint:60", "kind": "checkpoint", "path": "/Users/caoyuan/MyWork/Projects/Creator+/retool/retool-coding/retool-coding-0812/artifacts/training/checkpoints/retool-coding-0812-c2-seed42-step-60.json", "stage": "c2", "step": 60}], "observed_at": "2026-08-12T11:45:24.318870+00:00", "stage": "grpo-parallel"}`

### 2026-08-12T12:30:19.751507+00:00 — 01_train supervisor event

- `{"kind": "milestones", "milestones": [{"id": "c1:checkpoint:80", "kind": "checkpoint", "path": "/Users/caoyuan/MyWork/Projects/Creator+/retool/retool-coding/retool-coding-0812/artifacts/training/checkpoints/retool-coding-0812-c1-seed42-step-80.json", "stage": "c1", "step": 80}], "observed_at": "2026-08-12T12:30:19.751135+00:00", "stage": "grpo-parallel"}`

### 2026-08-12T13:17:36.056651+00:00 — 01_train supervisor event

- `{"kind": "milestones", "milestones": [{"id": "c1:checkpoint:100", "kind": "checkpoint", "path": "/Users/caoyuan/MyWork/Projects/Creator+/retool/retool-coding/retool-coding-0812/artifacts/training/checkpoints/retool-coding-0812-c1-seed42-step-100.json", "stage": "c1", "step": 100}, {"id": "c2:checkpoint:80", "kind": "checkpoint", "path": "/Users/caoyuan/MyWork/Projects/Creator+/retool/retool-coding/retool-coding-0812/artifacts/training/checkpoints/retool-coding-0812-c2-seed42-step-80.json", "stage": "c2", "step": 80}], "observed_at": "2026-08-12T13:17:36.056494+00:00", "stage": "grpo-parallel"}`

### 2026-08-12T13:49:00.073402+00:00 — 01_train supervisor event

- `{"kind": "complete", "observed_at": "2026-08-12T13:49:00.073234+00:00", "stage": "complete", "step": 100}`

### 2026-08-12 — Checkpoint-dev 配置校验阻塞

- 01_train 已验收完成：shared SFT 完成，C0/C1/C2 均到 step 100；三条分支的 step-20 early gate 均通过，监督器无错误、无自动恢复，训练进程以 exit code 0 退出。
- 已从 15 个 seed-42 checkpoint 生成 `artifacts/training/checkpoints/models.json`，并生成 C0/C1/C2 的冻结 checkpoint-dev 配置。
- 并行执行 `02_checkpoint_dev/run.py --validate-only` 时，在提交任何 dev 任务前被 MyEval 插件发现阶段阻断：已安装 entry point 引用了不可导入模块 `retool_lcb_mini`（`ModuleNotFoundError`）。
- 决策：这属于停止型环境/配置异常，不重试、不启动 checkpoint-dev 正式运行，也未启动 `03_evaluation`。待修复该 Python 包或移除失效 entry point 后，重新运行 checkpoint-dev 校验并再启动正式 dev。

### 2026-08-12 — Checkpoint-dev 启动与监督

- 复核后，MyEval 现已将失效的 `retool-lcb-mini` entry point 降级为 warning；C0/C1/C2 三份冻结配置的 `--validate-only` 均通过（各 5 个 checkpoint、3 个测试 prompt、100 个 dev task）。
- 运行时预检发现本机 Colima Docker 服务可用，但所需的原生 arm64 镜像 `retool-coding-0812-sandbox:py311` 缺失；已按项目 `docker/Dockerfile` 构建，镜像 ID 为 `sha256:62def2762600587bb0ff10108585eaf181b53aee394f11c7d5be635232eda023`。
- 已以 `02_checkpoint_dev/run.py --resume` 启动 C0/C1/C2；每个分支建立了可恢复的 SQLite run（每支 1,500 个 task），但本轮受监督会话的进程生命周期中断，尚未有 task 开始或完成，因而没有 checkpoint 选择结果。
- 边界：未启动或生成 `03_evaluation` 任何产物。下一次应从当前目录执行 `PYTHONPATH=src .venv/bin/python 02_checkpoint_dev/run.py --resume`，待三个 MyEval run 全部完成后才运行 `select_checkpoints.py`。

### 2026-08-12 — Checkpoint-dev 恢复未执行任务

- 在用户授权后执行 `PYTHONPATH=src .venv/bin/python 02_checkpoint_dev/run.py --resume`；命令退出码为 0，仅输出已知的失效 `retool-lcb-mini` entry point warning。
- 权威 run manifest 仍为 `running`：C0 有三个遗留 run、C1/C2 各一个；五个 SQLite 状态库均未产生 completed 或 failed task，已有的四个可读状态库各保持 `pending=1500`。因此本次没有生成 dev 分数或 checkpoint 选择结果。
- 该命令静默退出且未执行任务，属于需要进一步定位的停止型 MyEval 调度/恢复异常；未重试，未运行 `select_checkpoints.py`，也未启动或生成 `03_evaluation` 产物。

### 2026-08-12 — Checkpoint-dev 状态库膨胀根因与修复

- 根因：冻结 checkpoint-dev 数据中的个别 `private_tests` 单题序列化后超过 31 MB；MyEval 又把完整 sample 复制到每个 `模型 × prompt` task，导致每支仅 1,500 个 pending task 就写出约 16 GiB SQLite。C0 缺少分支互斥锁并被重复启动三次，因此五个 0-attempt/0-result 旧 run 共占约 81 GiB；建库阶段尚未创建 checkpoint-dev SwanLab run，也尚未调用 PyTRIO sampler。
- 修复：MyEval 状态层新增 sample 去重表并保持旧库读取兼容；本实验插件只在通用 EvalSample 中持久化 sample id，隐藏测试由 executor 从冻结文件按 id 懒加载；checkpoint-dev 与 final evaluation 新增分支级非阻塞文件锁，阻止同一 shard 重复启动。
- 规模验证：每个分支仍准确生成 1,500 tasks / 100 unique samples，新的 SQLite 为约 7.6 MB；三分支合计约 22 MB。项目 18 个测试、MyEval core/e2e/agent 30 个测试、三份 checkpoint-dev `--validate-only` 全部通过。
- 清理边界：五个旧 run 均为 `pending=1500`、`max_attempts=0`、`result_json=0`。删除约 81 GiB 的不可恢复操作未获安全授权，目录暂时保留；清理并重新运行 checkpoint-dev 前仍不得执行 checkpoint 选择或 final evaluation。

### 2026-08-12 — 清理旧 Checkpoint-dev 空状态

- 经用户明确授权，删除上述 5 个旧 run；删除前再次确认合计 7,500 个 task 全部为 pending、0 次尝试、0 个结果。
- `artifacts/checkpoint_dev` 的文件占用由约 81 GiB 降至 4 KiB；仅保留 C0/C1/C2 分支锁文件与 supervisor 的空日志/PID 记录。
- 删除不可恢复；其中没有模型输出、评分、SwanLab run 或 PyTRIO sampling 结果。修复后的 checkpoint-dev 尚未启动。

### 2026-08-13 — Checkpoint-dev 完成与 checkpoint 选择

- 修复后的 C0/C1/C2 正式 run 各包含 1,500 tasks（5 checkpoints × 3 prompts × 100 problems）。首轮 C2 完成；C0/C1 因 `abc350_e` 的暂时性 Docker exit 139 分别保留 5/1 个 pending，未产生 failed。
- 使用同一冻结 config/code fingerprint 执行 `PYTHONPATH=src .venv/bin/python -u 02_checkpoint_dev/run.py --resume`；恢复仅处理 6 个 pending，最终三支 manifest 均为 `complete`，SQLite 均为 `completed=1500, pending=running=failed=0`。
- SwanLab 在线验收均为 `FINISHED`：C0 `ckk2zdj0`、C1 `6fuljxi4`、C2 `gaplx8d5`。运行环境为 `pytrio==0.2.5`、`swanlab==0.9.2`。
- `02_checkpoint_dev/select_checkpoints.py` 的完整性与 paired-matrix 门禁通过。补充 15 个候选的完整明细时发现，C0 step 40/80 的 Average pass@1 都是精确的 `170/300`，原选择器受一 ULP 浮点表示差异影响误选 step 80；选择器已改为按整数通过数构造精确分数并增加回归测试。重新生成 `artifacts/checkpoint_dev/selected-models.json` 后，C0 正确选择 step 40，C1/C2 选择 step 100；详细分数与选择依据见上表。
- checkpoint-dev 全程由用户级 launchd `com.openai.retool-checkpoint-dev.caffeinate` 持有 `caffeinate -dims` 断言；启用时间 2026-08-13 00:50:38 +0800，验收完成后于 2026-08-13 06:04:56 +0800 卸载服务。`caffeinate` 进程及其 PreventUserIdleSystemSleep/DisplaySleep、PreventSystemSleep、PreventDiskIdle 断言均已确认消失。
- 边界：未启动 `03_evaluation`，也未生成 final evaluation 结果。

### 2026-08-13 — Final evaluation 完成

- C0/C1/C2 prompt shard 的同一批可恢复 MyEval run 均已完成：各 `800/800` completed，`pending=running=failed=0`，总计 `2,400/2,400`。
- 已运行 `03_evaluation/summarize.py`，最终矩阵归档为 [`docs/results/final-evaluation-matrix.md`](results/final-evaluation-matrix.md)；12 个 cells 均为 200 个配对的 temporally held-out test tasks，完整性、配对性与统计门禁均通过。
- 最终结果：C0/C1/C2 的跨 prompt Average pass@1 分别为 `0.338` / `0.382` / `0.380`，Worst 为 `0.315` / `0.375` / `0.360`；Base Average 为 `0.248`。全部 9 个已训练模型相对 Base 的 paired delta 为正，95% CI 不跨 0。

## 2026-08-13 — Post-hoc evaluation: C0-step100 与 shared-SFT-only

> Seed 42; the original 200 temporally held-out tasks; greedy decoding. This is a post-hoc checkpoint/SFT ablation and does not replace the preregistered selected-checkpoint final.

| Model | Test C0 | Test C1 | Test C2 | Average | Worst |
|---|---:|---:|---:|---:|---:|
| c0-step100 | 0.335 | 0.325 | 0.285 | 0.315 | 0.285 |
| shared-sft-only | 0.255 | 0.250 | 0.265 | 0.257 | 0.250 |

## Paired comparisons on the same 200 tasks

| Contrast | Test | Delta | Wins/Ties/Losses |
|---|---|---:|---:|
| c0-step100 - selected C0-step40 | C0 | -0.025 | 10/175/15 |
| c0-step100 - selected C0-step40 | C1 | -0.015 | 13/171/16 |
| c0-step100 - selected C0-step40 | C2 | -0.030 | 11/172/17 |
| shared-sft-only - raw Base | C0 | +0.005 | 14/173/13 |
| shared-sft-only - raw Base | C1 | +0.010 | 12/178/10 |
| shared-sft-only - raw Base | C2 | +0.010 | 15/172/13 |

## Interpretation boundary

- C0-step100 isolates the checkpoint-step sensitivity of the C0 branch; it is not a replacement checkpoint selected after seeing final results.
- shared-SFT-only separates the common neutral SFT contribution from the subsequent prompt-conditioned GRPO branches.
- Both additions are post-hoc and reuse the original test set, so they refine mechanism attribution rather than constitute a new confirmatory test.

## Provenance

- C0: `artifacts/evaluation-posthoc/c0/20260813T132142Z-retool-coding-0812-posthoc-c0-seed42-9cd108c3`
- C1: `artifacts/evaluation-posthoc/c1/20260813T132142Z-retool-coding-0812-posthoc-c1-seed42-8c775c4b`
- C2: `artifacts/evaluation-posthoc/c2/20260813T132142Z-retool-coding-0812-posthoc-c2-seed42-50526731`
