# 0812实验最终总结

> 实验：`retool-coding-0812`；模型：`Qwen/Qwen3.5-4B`；随机种子（seed）为 42。
> 配置选择见同级目录 `../07-retool-lcb-mini/`，正式训练、checkpoint-dev（检查点选择）与 evaluation（最终评测）见本目录。
> 最终评测使用 200 道按时间留出、不参与训练和配置选择的 LiveCodeBench v6（LCB-v6）题目；这是单随机种子实验，不应视为官方“无数据污染”LiveCodeBench 分数。

## 1. 配置选择

配置选择的目标是：先排除“因长度上限而来不及提交代码”的测量混淆，再找到能稳定产生 GRPO 学习信号的最简单交互协议。为此，先修复了真实停止原因、每轮生成长度和触顶状态的记录；之后的配置不只看正确率，还必须同时通过训练前门槛。

### 1.1 本节概念

- **Base**：未经本实验 SFT 或 GRPO 训练的 `Qwen/Qwen3.5-4B` 基础模型。
- **L2K、L4K、L8K**：生成长度配置的简写，最终的 **L10K/20K** 表示单轮上限 10,240、整条轨迹上限 20,480。
- **P1、P2**：工具交互协议。P1 允许整条轨迹最多 1 次 `run_python` 调用和 2 轮 assistant 回复（一轮可调用工具，一轮提交最终代码）；P2 放宽为最多 2 次工具调用和 3 轮 assistant 回复。两者的工具环境都是隔离、无状态的，每次工具返回最多 512 tokens。
- **shared SFT e1/e3/e4**：所有 GRPO 分支共享的中性监督微调（Supervised Fine-Tuning，SFT）初始状态；`e1/e3/e4` 表示同一批数据分别训练 1/3/4 轮（epoch）。

### 1.2 配置选择标准

**Hard gate（硬门槛）** 表示下表条件必须全部满足；任意一项失败就不允许启动正式 GRPO 训练。除基础设施错误外，其他门槛都要求 C0、C1、C2 三种 prompt 分别达标，不能用平均值掩盖某一分支的失败。

| 门槛 | 判定标准 | 它排除的问题 |
|---|---|---|
| 长度触顶门槛（cap gate） | token cap-hit 率 ≤5% | 回答因命中生成上限而被截断 |
| 格式门槛 | format-valid 率 ≥90% | 未按要求提交唯一 Python 代码块 |
| 工具门槛 | 存在工具调用尝试时，tool-valid 率 ≥80% | 已尝试的工具调用中，参数或协议不合法 |
| 学习信号门槛 | nondegenerate group 率 ≥40% | 同题 8 条 rollout 的 reward 完全相同，无法产生组内相对优势信号 |
| 基础设施门槛 | infrastructure error = 0 | Docker、judge（判题器）或运行环境故障污染实验结果 |

选择顺序是：先看是否通过全部硬门槛；通过后再选改动最小、交互最简单、成本更低的配置。在早期所有长度候选都未达标时，才用“格式失败更少、pass@1 更高、长度更短”作为临时回退规则；这类临时候选不等于已可进入训练。

| 阶段 | 比较内容 | 定性结论与决策 |
|---|---|---|
| Base 长度消融 | 固定 P1，比较 L2K、L4K、L8K | 三种长度均未解决截断和格式问题；L4K 只是按回退规则暂选，不具备训练资格。 |
| 工具协议消融 | 在 L4K 下比较 P1 和 P2 | P2 增加了一次工具调用和一轮交互，但正确率没有改善，截断和格式反而更差；保留 P1。 |
| Base + L4K/P1 训练前小规模检查（smoke） | 在 C0/C1/C2 上做小规模训练前测试 | 硬门槛失败，说明只改长度和交互预算不够；加入所有分支共享的中性 SFT 初始化。 |
| SFT 轮数与长度复核 | 比较 e1/e3/e4，并从 L4K 逐步扩到 L8K | e3 已修复大部分协议行为，但仍需略微扩长；e4 虽改善了截断，却破坏了 C1 的学习信号，因此不选“更多 SFT”。 |
| **e3 + L10K/20K + P1** | 保持 e3、P1 和 reward 不变，只小幅扩大长度 | C0/C1/C2 同时通过全部硬门槛；这是第一个达标的最小改动，因此选为正式训练配置。 |

最终配置为 **shared SFT e3 + L10K/20K + P1**，即共享中性 SFT 训练 3 轮，单轮 assistant 最多 10,240 tokens，整条轨迹最多 20,480 tokens，最多一次工具调用和两轮 assistant 回复。该配置的 C0/C1/C2 都通过了上述五项硬门槛。数据固定为 400 道训练题、60 道配置校准题、100 道 checkpoint 选择题和 200 道最终测试题；shared SFT 使用 300 个不同题目的中性正确轨迹。

## 2. Training

### 2.1 实际 system prompt

System prompt 是在题目之前提供给模型的系统级行为指令。C0/C1/C2 共用以下完整基础 prompt：C0 不再追加内容，C1 追加“简洁思考并优先提交”的要求，C2 追加六步结构化解题流程。

```text
You are an expert Python competitive-programming agent. Solve the user's problem with a correct and efficient Python 3.11 program using only the standard library. Follow the supplied starter code when present; otherwise read from stdin and write to stdout.

You may call run_python to execute code with optional stdin. Each call is isolated and stateless. Call it at most once per assistant turn and wait for the result.

When the solution is ready, return exactly one fenced Python code block and no other text.

You may call run_python at most 1 times in total. A tool call must be a short, focused check rather than a full solution or a reasoning scratchpad; omit long comments and exploratory code. When the tool budget is exhausted, or when further testing is unnecessary, immediately submit the final code in the required format.
```

C1 在基础 prompt 后追加：

```text
Reason concisely about the input/output, constraints, algorithm, and edge cases, then prioritize submitting the final implementation directly. Use run_python only for one short, focused check that resolves a specific uncertainty; never use it to draft the full solution or as a reasoning scratchpad. Reserve most of the response budget for the final program.
```

C2 在基础 prompt 后追加：

```text
Follow this workflow:

1. Identify the required input/output behavior, constraints, and important edge cases.
2. Derive a correct algorithm and verify its time and space complexity.
3. Implement the algorithm carefully in the required starter-code or stdin/stdout format.
4. Use run_python to check the provided examples when useful.
5. Check important boundary or adversarial cases, and fix the code if a check fails.
6. Review correctness, complexity, and output formatting before submitting the final code.
```

### 2.2 重要训练配置

| 项目 | Shared-neutral SFT | C0/C1/C2 GRPO |
|---|---|---|
| 基础模型 / LoRA | `Qwen/Qwen3.5-4B` / rank 32 | 从同一 SFT 初态分别训练；rank 32 |
| 数据 | 300 条中性正确轨迹 | 400 道训练题；每分支固定顺序、每题恰好使用一次 |
| 规模 | 3 轮，114 个优化器更新 step | 每分支 100 steps；每步 4 题 × 每题 8 条 rollout = 32 条轨迹；每分支 3,200 条 |
| 采样 | — | temperature（概率分布平滑系数）为 1.0，top-p（累计概率候选集阈值）为 1.0；即按模型原分布做随机采样，不另外截断低概率候选集 |
| 损失函数 | 交叉熵；仅计算 assistant 回复；手工自回归错位 | 重要性采样；同题 advantage = 当前 reward − 组内平均 reward |
| Reward（奖励） | — | 格式非法为 `-0.1`；格式合法时为测试用例通过比例；全部通过为 `1.0` |
| 优化器 | Adam；学习率 `1e-5`；动量系数 β=`0.9/0.95` | Adam；学习率 `4e-5`；动量系数 β=`0.9/0.95` |
| 批次限制 | 最多 8 datums；padded tokens ≤65,536；context 12,288 | 最多 16 datums；padded tokens ≤65,536 |
| 保存与训练期门槛 | 每 20 steps 保存完整 state | step 20/40/60/80/100 保存；step 20 要求 nondegenerate group 累计比例 ≥50%、跳过参数更新的比例 ≤25% |
| 协议与长度 | — | P1；assistant 10,240；trajectory 20,480；tool response 512 |
| 运行环境 | `pytrio==0.2.5`（远程模型训练和采样的软件开发工具包，SDK）；`swanlab==0.9.2` | 同左；判题环境为 Python 3.11、断网、只读根目录、1 个中央处理器（CPU）核、512 MiB 内存（MiB 为 2^20 字节）、最多 64 个进程 |
| 执行超时 | — | 工具调用 5 秒；单个测试用例 3 秒；整题判题 30 秒 |

这个 step-20 门槛是训练期的早停检查，与第 1 节的训练前硬门槛不同。三条 GRPO 分支均通过 step-20 检查：C0/C1 的 nondegenerate group 累计比例为 `77.5%`，C2 为 `82.5%`，跳过参数更新的比例均为 `0%`；随后全部训练至 step 100。

### 2.3 SwanLab 训练曲线

SwanLab 是本实验的训练记录与曲线可视化平台。以下运行链接已在线核验，状态均为 `FINISHED`（已完成）：

- [Shared-neutral SFT（114 steps）](https://swanlab.cn/@tonycaoyuan/retool-coding-0812/runs/chjlx0h7/chart)
- [C0 GRPO（100 steps）](https://swanlab.cn/@tonycaoyuan/retool-coding-0812/runs/a71cec9q/chart)
- [C1 GRPO（100 steps）](https://swanlab.cn/@tonycaoyuan/retool-coding-0812/runs/eyqu4mod/chart)
- [C2 GRPO（100 steps）](https://swanlab.cn/@tonycaoyuan/retool-coding-0812/runs/2hggeal6/chart)

## 3. Checkpoint-dev

Checkpoint-dev 是“用开发集选择检查点”的阶段；这 100 道开发题只用于选 step，不是第 4 节的最终测试集。每个分支的 step 20/40/60/80/100 都用 C0/C1/C2 三种测试 prompt 做 greedy 评测（确定性解码；每题只生成一个答案）。每行含 300 条轨迹，合计 4,500 条全部完成，无评测任务失败。

`pass@1` 表示每题只生成一个答案时，能通过该题全部测试用例的题目比例。`case-pass` 表示所有单个测试用例的平均通过比例。检查点选择顺序为：先比较三种测试 prompt 的平均 pass@1，再比较三者中的最低 pass@1，然后比较平均 case-pass，仍持平时选更早的 step。

| 训练分支 | 训练 step | C0 prompt | C1 prompt | C2 prompt | 三 prompt 平均 pass@1 | 三 prompt 最低 pass@1 | 平均 case-pass |
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

最终选择：

- **C0 step 40**：与 step 80 的三 prompt 平均 pass@1 相同，但三个 prompt 中的最低值更高。
- **C1 step 100**：本分支的三 prompt 平均 pass@1 最高。
- **C2 step 100**：本分支的三 prompt 平均 pass@1 最高，也是全部候选中最高的。

值得注意的是，C0 很早进入平台；C2 则明显“晚熟”，step 40 一度退化，step 100 才同时改善整题正确率、测试用例通过率、输出格式和长度触顶率。

## 4. Evaluation

最终评测（evaluation）在同一批 200 道留出测试题上，用 C0/C1/C2 三种测试 prompt 评估不同模型权重。全部结果均使用确定性解码，即每道题只生成一个答案，不使用随机采样。

原计划的最终评测包含 raw Base（未微调的原始基础模型）、shared-SFT-only 和三个入选 checkpoint，共 15 个“模型权重 × 测试 prompt”组合、3,000 条轨迹。为了更直观地展示“C0 选择了 step40 而非 step100”的依据，又补充了 C0 step 100 事后分析（post-hoc，即看到原最终结果后追加），增加 3 个组合、600 条轨迹。下表合并展示 18 个组合、3,600 条轨迹；全部完成，无评测任务失败。

### 4.1 Train × Test（训练来源 × 测试提示词）pass@1

本表中，行表示“模型权重的训练来源”，列表示“推理时使用的测试 prompt”。“三 prompt 平均”是 C0/C1/C2 三列的平均值；“三 prompt 最低值”是三列中最差的一列，衡量跨 prompt 稳健性；“同名 prompt 差值”是训练分支在同名测试 prompt 上的 pass@1，减去其余两种测试 prompt 的平均 pass@1。没有 C0/C1/C2 训练分支名的 Base 和 SFT-only 不计算该差值。

| 模型权重（训练来源） \ 测试 prompt | C0 | C1 | C2 | 三 prompt 平均 | 三 prompt 最低值 | 同名 prompt 差值 |
|---|---:|---:|---:|---:|---:|---:|
| Raw Base | 25.0% | 24.0% | 25.5% | 24.83% | 24.0% | — |
| Shared-SFT-only | 25.5% | 25.0% | 26.5% | 25.67% | 25.0% | — |
| C0 step 40（入选） | 36.0% | 34.0% | 31.5% | 33.83% | 31.5% | +3.25 个百分点 |
| C1 step 100（入选） | 38.5% | 38.5% | 37.5% | **38.17%** | **37.5%** | +0.50 个百分点 |
| C2 step 100（入选） | **40.5%** | 36.0% | 37.5% | 38.00% | 36.0% | −0.75 个百分点 |
| C0 step 100（事后分析） | 33.5% | 32.5% | 28.5% | 31.50% | 28.5% | +3.00 个百分点 |



为了判断入选模型的提升是否只由少数题目偶然造成，又将每个入选模型与 raw Base 在**同一道题**上直接比较，这就是“逐题配对比较”。下表中：

- `pp` 是 percentage points（百分点），例如从 25% 到 36% 是提升 11 pp。
- 方括号是通过 10,000 次逐题 bootstrap（重复抽样）得到的 95% 置信区间（confidence interval，CI）；区间不包含 0，表示在该不确定性分析下，提升方向较稳定。
- Exact McNemar 检验专门比较同一批题上两个模型的“对/错”结果。因为这里同时检查 9 个组合，又用 Holm 校正调严多次比较的判定标准，减少“比较次数越多，越容易偶然报出差异”的问题。这里的 `0.05` 是统计判定门槛，不是正确率；达到该门槛表示在“两个模型没有差异”的假设下，当前逐题差异不太像只是随机题目组成造成的。

结果是：三个入选模型在三种测试 prompt 下，相对 raw Base 的 9 个 pass@1 差值全部为正，且 95% 置信区间下界都高于 0；经 Holm 校正后，9 项比较仍全部达到预设的 0.05 显著性门槛。

| 训练分支 \ 测试 prompt | C0 | C1 | C2 |
|---|---:|---:|---:|
| C0 | `+11.0pp [5.5,17.0]` | `+10.0pp [4.0,16.0]` | `+6.0pp [0.5,11.5]` |
| C1 | `+13.5pp [7.5,19.5]` | `+14.5pp [8.5,20.5]` | `+12.0pp [6.5,17.5]` |
| C2 | `+15.5pp [10.0,21.0]` | `+12.0pp [7.0,17.5]` | `+12.0pp [6.0,18.5]` |

单个组合最高为 **C2 训练 × C0 测试 = 40.5%**；三种测试 prompt 的平均值和最低值则都是 **C1 最高，分别为 38.17% 和 37.5%**。

### 4.2 最值得注意的其他数据

下表把每个模型在三种测试 prompt 上的 600 条轨迹聚合在一起。`C0@40` 表示 C0 训练分支的 step 40 checkpoint，其他 `@` 写法同理。指标含义如下：`case-pass` 是单个测试用例的通过率；`format-valid` 是严格输出唯一 Python 代码块的比例；`cap-hit` 是回答触发生成长度上限的比例；“合法格式中整题通过”是只在 format-valid 回答中计算的 pass@1；“平均生成 tokens”不包含输入 prompt；“平均时延”是一条完整轨迹的墙钟时间。箭头表示希望方向。

| 指标 | Raw Base | Shared SFT only | C0@40 | C0@100(*) | C1@100 | C2@100 |
|---|---:|---:|---:|---:|---:|---:|
| 整题 pass@1（↑） | 24.83% | 25.67% | 33.83% | 31.50% | **38.17%** | 38.00% |
| 测试用例 case-pass（↑） | 35.36% | 37.80% | **48.05%** | 43.35% | 47.65% | 46.99% |
| 输出格式 format-valid（↑） | 63.33% | 69.50% | **84.00%** | 73.33% | 65.00% | 69.67% |
| 长度触顶 cap-hit（↓） | 37.67% | 32.33% | **16.67%** | 27.00% | 35.83% | 30.50% |
| 合法格式中整题通过（↑） | 39.21% | 36.93% | 40.28% | 42.95% | **58.72%** | 54.55% |
| 平均生成 tokens（↓） | 4,563 | 4,195 | 3,643 | 4,542 | 5,624 | 4,880 |
| 平均时延（↓） | 41.3 秒 | 78.6 秒 | 57.5 秒 | 86.4 秒 | 86.9 秒 | 77.7 秒 |

`*` 表示事后追加的机制消融（只改变一个组件来判断收益来源），用于解释原最终结果，不替代原计划的最终评测。
时延不只由生成 token 数决定，还包括工具调用、代码执行和判题时间，因此它不会与生成长度严格同步变化。

从这些指标可以看出四个不同的机制：

1. **Shared SFT 改善了提交行为，但没有明显提高整题正确率。** SFT-only 比 raw Base 少触顶、格式更合法，但平均 pass@1 只高 0.83 个百分点，95% 置信区间为 `[−2.33, 4.17]`，包含 0；合法格式中的整题通过率甚至没有提高。这说明 shared SFT 的主要作用是建立可训练的协议起点，而不是单独解决算法正确性。
2. **C0 step 40 的优势主要是“能及时交付”。** 它的格式合法率最高、长度触顶率最低，在三个 GRPO 入选模型中生成最短；但只看格式合法的答案，其整题通过率明显低于 C1/C2。因此 C0 更像是学会了收敛、调用工具和提交，而不是在合法代码的算法质量上最强。
3. **C1/C2 的优势更多来自合法代码的正确性。** 两者都仍有较多触顶，但一旦成功提交合法格式，整题通过率显著高于 C0。C1 的该指标最高，且三种测试 prompt 中的最低 pass@1 也最高；C2 的总体 pass@1 几乎相同，但平均少生成 745 tokens、快约 9.2 秒。
4. **C0 继续训练后的主要问题是终止退化。** step 100 与入选的 step 40 相比，平均 pass@1 低 2.33 个百分点，95% 置信区间为 `[−5.50, 0.83]`；同时格式合法率下降、触顶率上升、生成变长。它在合法格式中的整题通过率略高，但这一小幅改善被“更多回答无法及时提交”完全抵消。统一比较 step 100 时，C1/C2 的平均 pass@1 仍分别比 C0 高 6.67/6.50 个百分点，因此 C0 较弱不能仅归因于选了更早的 checkpoint。

## 5. 数据分析与结论

### 5.1 收益来源：GRPO 而非 shared SFT

Shared-SFT-only 与 raw Base 的平均 pass@1 基本持平，但格式和长度行为更好，说明这一阶段主要负责建立一个“会按协议交付”的共同起点。相对同一 SFT 初态，C0、C1、C2 经 GRPO 后的平均 pass@1 分别提高约 8.17、12.50 和 12.33 个百分点。结合三个入选模型相对 raw Base 在 9 个正式比较中都有正向提升，可以将主要的正确率收益归因于后续 GRPO，而不是 shared SFT 本身。

这一归因仍有边界：SFT-only 是事后追加的分析，而且只有一个随机种子；它足以解释当前这次实验，但还不足以宣称在其他数据切分和随机种子上必然重复。

### 5.2 训练 prompt 比测试 prompt 更重要

“训练 prompt”决定 GRPO 期间如何生成 rollout、获得 reward 并更新权重；“测试 prompt”只是在权重固定后改变推理时指令。六种模型状态的三 prompt 平均 pass@1 从 24.83% 到 38.17%，跨度为 13.33 个百分点；反过来，固定模型权重，只把测试 prompt 从 C0/C1/C2 之间切换，六个模型整体平均的最大差距只有 2.00 个百分点。这表明长期的权重行为差异比推理时多加一段说明更重要。

**更详细的 test prompt 仍没有稳定收益，并且没有可靠的 train/test 同名匹配收益。** 三个入选分支在同名 prompt 组合上的平均 pass@1 为 37.33%，其他交叉组合为 36.33%，差值仅 +1.00 个百分点，95% 置信区间为 `[−1.08, 3.17]`，包含 0。因此推理时默认用最短、最简单的 C0 prompt 即可；没有必要为了“训测同名”增加 prompt 长度。

### 5.3 不同分支与 checkpoint 学到了不同行为

- **C0 step 40** 是终止和提交行为最好的分支：格式合法率高、触顶少、生成短，但合法代码的整题正确率低于 C1/C2。
- **C1 step 100** 的三 prompt 平均 pass@1 和最低 pass@1 都最高，合法代码的整题正确率也最高，适合优先考虑跨 prompt 稳健性的场景；代价是生成最长、平均时延最高。
- **C2 step 100** 与 C1 的三 prompt 平均 pass@1 只差 0.17 个百分点，并产生全表最高的单格结果（C2 训练 × C0 测试）；同时它比 C1 少生成 745 tokens、快约 9.2 秒，是当前更好的质量—成本折中。

Checkpoint 选择不能默认取最后一步。C0 从 step 40 继续训练到 step 100 后，生成更长、格式更差、触顶更多，最终 pass@1 反而下降；C2 则到 step 100 才明显改善。两者方向相反，说明必须使用固定开发集、固定的跨 prompt 规则选择 checkpoint，不能用“训练步数更多应该更好”替代实测。

### 5.4 最大剩余问题是不能及时收束

合并 6 种模型状态的 3,600 条轨迹后，1,051 次格式失败全部同时命中长度上限；1,080 条触顶轨迹中没有一条通过整题。因此 cap-hit 不是一个轻微的效率指标，而是几乎直接对应失败的行为指标。

继续单纯扩大 token 上限可能降低表面的触顶率，但会增加训练和推理成本，也可能掩盖“模型没有在代码已可提交时及时结束”的根本问题。下一步更值得测试的是 termination-aware reward（显式考虑是否在上限前完成合法提交的终止感知奖励），例如对触顶单独惩罚、对及时合法提交给予小额奖励，并与当前 reward 做严格消融；重点应是改善 C1/C2 的终止行为，而不是继续加长测试 prompt。

## 6. 后续实验路线图

后续工作建议分为三类：先用现有 checkpoint 做低成本机制诊断，再用 20–40 step canary 筛选训练改动，最后只对胜出的方案重跑完整多 seed 流程。这样可以分别回答“为什么 cap-hit”“工具究竟有没有增益”“现有结论能否跨随机种子复现”，同时避免一次完整训练同时改变多个变量。

### 6.1 对几个直接建议的判断

| 建议 | 判断 | 更合适的做法 |
|---|---|---|
| 比较 P1、P2 和更多工具调用规定 | **值得做，但不建议直接把 P2 全量重跑。** 之前 Base + L4K 的同题校准中，P2 相对 P1 在三个 prompt 下 pass@1 都低 4pp，cap-hit 高 5.3pp、格式失败高 8.0pp；现有证据不支持把“更多调用次数”当作默认改进。另一方面，当前 trained model 几乎每题都尝试工具，却没有干净的无工具对照，因此协议消融仍有重要缺口。 | 优先补 P0（无工具）与 P1 的对照；P2 先做固定 checkpoint 的推理诊断，只有出现明确的第二次调用增益才进入训练。协议编号 P0/P1/P2 与 system prompt 编号 C0/C1/C2 必须在报告中始终分开。 |
| 尝试更多 system prompt 降低 cap-hit | **只适合做针对性 prompt，不适合继续枚举“更长、更详细”的通用 prompt。** 六个模型的 test-time C0/C1/C2 已经说明普通详细化没有稳定收益。 | 只改变与终止直接相关的轴，例如“预留 final submission 预算”“工具返回后立即收束”“剩余预算提醒”；先在现有 C1/C2 checkpoint 上评测，再决定是否把同一规则放入训练。 |
| 用其他方法降低 cap-hit 后重跑完整流程 | **方向正确，但必须先区分长度不足、未学会终止和上游解题失败。** 直接增加最大 token 或增加 cap penalty 都可能只让模型更晚截断或更早提交错误代码。 | 先做长度扩展、强制收束和轨迹分类三类诊断；再比较 termination-aware 训练。只有同时降低 cap-hit、保持或提高 pass@1 的方案才进入完整流程。 |
| 更换 seed 重跑完整流程 | **必要，且是当前最高优先级的 confirmatory 实验。** 单个新增 seed 能发现明显不稳定，但仍不足以支持稳健结论。 | 至少做到 3 个总 seed（现有 42，加 43、44），理想为 5 个；使用未参与本轮分析的 fresh test，并在看结果前冻结主比较和 checkpoint 规则。 |

### 6.2 优先级 A：先拆解 cap-hit 的成因

当前 3,600 条 evaluation 轨迹中，cap-hit 下通过率为 0，全部格式失败都伴随 cap-hit；但这仍不能区分“再给一些 token 就能完成”和“模型进入了不收束的生成模式”。建议先固定现有 C1/C2 checkpoint 和 test C0，做以下配对诊断：

1. **长度反事实：L10K vs L16K。** 在现有 44 道跨模型高频 cap 题上做机制分析，并另取一组未用于挑选条件的 fresh、按难度分层任务估计总体效应。若 L16K 能把大量 cap 轨迹转为合法且正确的提交，说明仍有真实预算不足；若只是继续变长或仍然无效，则不应再靠扩容解决。
2. **强制收束反事实。** 保持总预算不变，在约 8K token 或工具返回后触发一次“停止分析并提交最终代码”的 finalization cue，并为最终代码预留固定 token。比较它与纯 prompt 版终止提醒，区分运行时预算控制和静态文字提示的作用。
3. **cap 轨迹分型。** 对截断轨迹自动统计并抽样盲审：工具调用过晚、工具后继续发散、重复推导、反复改写代码、已经有可用代码但未提交、直到上限仍无可用实现。各类占比决定下一步应优化终止、工具时机还是算法能力。

这里的主指标不应只是 cap-hit rate，还应同时报告 `pass@1`、format-valid、`Pass given valid format`、completion tokens 和 latency。原 44 题属于 post-hoc 机制样本，不能用其恢复率代替 fresh set 上的总体收益。

### 6.3 优先级 A：termination-aware 训练消融

静态 test prompt 已显示弱效应，因此如果强制收束诊断有效，下一步应验证训练是否能把该行为内化。建议固定更有成本优势的 C2 策略，先做一个小型、词数尽量匹配的消融：

| 分支 | 改动 | 目的 |
|---|---|---|
| T0 | 当前 C2 + 当前 reward | 复现基线 |
| T1 | 只增加明确的终止/预算规则 | 测 prompt 中的终止语义能否在训练中内化 |
| T2 | 当前 prompt + termination-aware shaping | 测奖励信号是否比文字提示有效 |
| T3 | 终止规则 + shaping | 检查二者是否互补 |

shaping 不宜只奖励“更短”。更安全的候选是：对 cap-hit 施加有限惩罚；只对**合法 final submission**提供小额完成奖励；或在正确性相同的轨迹之间才偏好更早提交。否则模型可能通过快速输出格式合法但错误的代码获得代理奖励。先运行 20–40 steps，并预注册继续条件：相对 T0 的 cap-hit 明显下降、pass/case-pass 不下降、format-valid 上升，且 nondegenerate group 与 skipped update 仍通过训练门槛。未通过者不跑到 step 100。

此外，可在固定 C2 主体下做 `termination cue × selective-tool cue` 的 2×2 小实验；它比继续添加任意风格的 C3/C4 prompt 更容易解释。为避免把“文本更短”误当作策略效应，应使用长度相近的中性文字作为 control。

### 6.4 优先级 A：工具协议的因果消融

工具协议建议分两阶段。第一阶段不训练，先把 Base、SFT-only 和现有 C2 checkpoint 在同一批 fresh 任务上交叉评测：

| Protocol | 规则 | 主要问题 |
|---|---|---|
| P0 | 无工具，单次 assistant 直接提交 | 工具环境相对纯推理到底增加多少能力 |
| P1 | 当前协议：最多 1 次调用、2 个 assistant turns | 当前基线 |
| P2 | 最多 2 次调用、3 个 assistant turns | 第二次调用能否用于修复第一次检查发现的问题 |

重点记录 `P1−P0` 的工具净收益和 `P2−P1` 的第二次调用边际收益，并把 P2 轨迹拆成“实际只用 0/1 次”和“确实使用第 2 次”两组。还应记录首次调用前 token、工具后 token、第二次调用是否改变最终通过状态，以及 cap 是发生在工具前还是工具后。已有 P2 校准方向偏负，所以如果预算有限，正式训练优先做 **P0 vs P1**；只有 P2 在 fresh 诊断中提高通过率且没有显著增加 cap/成本，才训练 P2。

若进入训练，应固定同一个 C2 system strategy、数据顺序、初始化、reward 和 token budget，只改变 protocol。P0/P1/P2 的整套训练可以回答协议容量问题，但由于 call 数与交互 turn 数天然联动，结论应表述为“整个交互协议的效果”，不要把差异单独归因于调用次数。

另一个较低优先级的协议实验，是比较 Qwen 原生 tool call 与论文的 `<code>/<interpreter>` 文本协议。它能回答收益是否依赖模型预训练熟悉的 schema，但应排在 P0/P1 和 cap 机制之后。

### 6.5 优先级 A：多 seed、fresh split 的完整复验

探索阶段结束后冻结唯一候选 recipe，再进行完整 confirmatory run。推荐最小设计如下：

- seed 使用 `42/43/44`，若预算允许扩到 5 个；一个新 seed 只能称为 replication pilot。
- 最低限度保留 C0 与“胜出的 C2-termination”两条 GRPO 分支；若预算足够再加入原 C1/C2，以复验 C1/C2 的成本—质量排序。
- 每个 seed 独立重训 shared SFT；同一 seed 内各分支共享初始化、题目顺序和采样随机流，seed 之间更换随机性。若更换训练题顺序，同一 seed 的各分支必须使用同一排列。
- 把**统一 step 100**设为 primary comparison，把预先冻结规则选出的 dev checkpoint 设为 secondary。当前 C0 step40/80 在 dev 精确同分、dev 到 final 又下降 23–26pp，说明不能只依赖一次 100 题 dev 排序。
- 使用从未参与 prompt、协议、reward 和 checkpoint 选择的 fresh test split。原 200 题继续可作回归集，但不再承担 confirmatory 显著性检验。
- 按 seed 分别报告效应和方向，再做跨 seed 汇总；不要把不同 seed 的所有轨迹简单当作独立样本。预注册 primary contrasts、Holm family、失败标准和“至少多少 seed 同方向”。

如果预算只允许一条新增完整流程，优先顺序应是：**先完成 seed 43 的当前冻结 C0/C2 复验，而不是训练更多 test prompt 或直接跑 P2**。但最终报告仍应明确它只有两个 seed，不能替代三到五 seed 的稳定性结论。

### 6.6 优先级 B：reward 与难度分布

当前 C0 的 case-pass 最高但完整通过率更低，说明“按通过 case 比例给连续奖励”可能偏好部分正确解；同时 hard 题通过率仍低于 11%。在 cap 问题得到初步控制后，可做两组进一步实验：

1. **reward 2×2：** dense case reward vs binary all-pass reward，分别有/无 termination shaping。binary reward 可能显著增加 degenerate group，因此必须把 nondegenerate rate 和 update-skipped 作为 early gate，而不能只看最终 pass。
2. **难度/失败类型课程：** 提高 hard 题、高 case-pass wrong-answer、runtime error 和 time-limit 样本的采样权重；训练集与评测集按题目严格隔离，并按 easy/medium/hard 分层报告。该实验用于解决语义正确性，不应与工具协议消融同时改动。

还可增加“工具是否必要”的离线标签或盲审标签，分析直接提交、有效一次检查、无效仪式化调用三类轨迹。若未来把该信号加入 reward，应先验证它不会诱导模型为了标签而规避有用工具。

### 6.7 推荐执行顺序

1. **零训练诊断：** cap 轨迹分型、L10K/L16K、强制收束、P0/P1/P2 fixed-checkpoint 评测。
2. **小规模筛选：** termination 2×2 canary；必要时再做 P0/P1 或 reward canary，统一在 20–40 steps 设停止门槛。
3. **完整复验：** 只选择一个胜出 recipe，补 seed 43/44、统一 step100 与 fresh test。
4. **扩展研究：** hard-task curriculum、reward 形式、原生 tool-call vs 文本协议、4B vs 更大基座。

这一路线中，最关键的新信息依次是：**cap-hit 能否被可靠转化为有效提交、工具相对无工具是否有真实净收益、改进是否跨 seed 和 fresh split 成立。** 在这三点解决前，继续增加任意 system prompt 数量或直接扩大工具调用上限，信息收益都相对有限。
