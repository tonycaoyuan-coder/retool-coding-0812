# 代码导航指南

## 一句话数据流

`冻结输入 → shared SFT → C0/C1/C2 GRPO → checkpoint-dev 选择 → 12-cell evaluation → 统计与报告`

## 运行入口

| 程序 | 用途 | 重要说明 |
|---|---|---|
| `01_train/run_all.py` | 先运行一次 shared SFT，再并行启动三个 GRPO 分支 | 任一子进程失败时会终止其余分支；用 `--resume` 从已提交状态恢复 |
| `01_train/shared_sft.py` | 训练三轮 assistant-only shared-neutral SFT | 手工自回归右移；每 20 step 保存含 optimizer 的完整远端 state |
| `01_train/run.py` | C0/C1/C2 的冻结参数入口 | 只接受 `--variant` 与恢复参数，防止命令行覆盖正式超参数 |
| `01_train/train_branch.py` | 单个 GRPO 分支的完整训练循环 | 负责 gate、rollout、group advantage、micro-batch、更新、checkpoint 与 SwanLab |
| `01_train/supervisor.py` | 单实例启动、状态快照和有限自动恢复 | 只恢复可重试故障；计费、认证、指纹或数据不一致会停止 |
| `02_checkpoint_dev/build_model_manifest.py` | 收集每个分支的 20/40/60/80/100 checkpoint | 缺失、重复或 seed 不符会拒绝继续 |
| `02_checkpoint_dev/select_checkpoints.py` | 按跨 prompt 规则选择每个训练分支的 checkpoint | 排序顺序为 average pass@1、worst pass@1、case-pass、较早 step |
| `03_evaluation/run.py` | 运行最终三路 prompt shard | 每路包含 Base 与 C0/C1/C2，共形成 12 个 cell |
| `03_evaluation/summarize.py` | 合并三路评测并计算配对统计 | 检查任务集合完整且可配对后才输出结果 |

## 核心共享模块

| 模块 | 负责什么 | 阅读时关注 |
|---|---|---|
| `src/retool_coding_0812/settings.py` | 加载并严格校验 `configs/experiment.yaml` | 所有正式参数都与冻结值逐项比较，避免配置漂移 |
| `src/retool_coding_0812/protocol.py` | C0/C1/C2 prompt、工具协议、消息拼接与输出解析 | 工具调用和最终代码的格式约束；多轮 token 拼接不能重复 chat-template 前缀 |
| `src/retool_coding_0812/rollout.py` | PyTRIO 并发采样、Docker 工具执行、多轮轨迹状态机 | old logprob 必须来自 rollout 时的 student；tool observation 进入上下文但不进入 loss |
| `src/retool_coding_0812/training_utils.py` | 把轨迹转换为 GRPO Datum、打包 micro-batch、汇总指标 | prompt/observation 的 target、logprob、advantage 均以零占位；completion 使用 group-relative advantage |
| `src/retool_coding_0812/sft_utils.py` | 校验正确轨迹并构造 assistant-only SFT Datum | 只有 assistant completion 权重为 1；输入、target 和 weight 都手工右移对齐 |
| `src/retool_coding_0812/docker_executor.py` | 隔离执行工具代码和最终判题 | 断网、只读根目录、CPU/内存/pid/超时限制 |
| `src/retool_coding_0812/gates.py` | 数据、镜像、prompt、工具和协议指纹门禁 | 任何来源或参数漂移都会在创建 TrainingClient 前失败 |
| `src/retool_coding_0812/resume.py` | checkpoint 发现和本地产物回滚 | 远端已保存的 optimizer state 是权威；更新但未 checkpoint 的本地产物移入 `recovery/` 而非删除 |
| `src/retool_coding_0812/myeval_plugin.py` | MyEval benchmark 与 agent executor 注册 | 评测复用训练时的 rollout、Docker 判题和行为指标定义 |

## 三个容易误改的不变量

1. Shared SFT 恢复时必须加载包含 optimizer 的 state；GRPO 从 shared SFT 初始化时必须丢弃 SFT 的 Adam moments。
2. GRPO 的 old logprob、target 和 advantage 必须与 `model_input` 右移后逐 token 对齐；prompt 和工具 observation 区间只能作为上下文。
3. checkpoint 后、下一次 checkpoint 前产生的 metrics/轨迹并不对应已保存的远端 optimizer state，恢复时必须回滚并确定性重跑。

## 本地检查

```bash
export PYTHONPATH=src
.venv/bin/python -m pytest
.venv/bin/python 01_train/run_all.py --gate-only
```

第二条会检查冻结配置、数据/协议指纹和 Docker 运行环境，但不会创建 PyTRIO 训练任务。
