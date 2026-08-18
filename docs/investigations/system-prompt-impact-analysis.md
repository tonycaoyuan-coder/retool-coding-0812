# ReTool-Coding 0812 实验：Train/Test System Prompt 影响分析

> 分析对象：`retool-coding-0812`，seed 42。
> 权威数据：本地训练 `metrics.jsonl` 与 9,600 条正式训练轨迹、4,500 条 checkpoint-dev 轨迹、3,600 条 evaluation 轨迹（2,400 条 selected-checkpoint final + 1,200 条 post-hoc final）。
> 统计更新时间：2026-08-14；两批 final 均使用同一批 200 道 temporally held-out LCB-v6 题、greedy decoding，合计 6 model × 3 test prompt 完整配对。
> 本报告沿用 ReTool 原实验的分析顺序：**运行健康度 → 训练过程 → checkpoint 选择 → final 主指标 → secondary metrics → 单题下钻**。

## 0. 结论先行

一句话结论：**训练时 system prompt 确实改变了模型学到的行为与最终能力，但推理时临时换成更详细的 system prompt 几乎没有稳定收益；C1/C2 的主要价值来自训练后内化的解题质量，而不是 test-time prompt matching。**

最重要的发现如下。

1. **训练分支效应远大于 test prompt 效应。** 跨 test prompt 平均后，Base / SFT-only / C0-step40 / C0-step100 / C1-step100 / C2-step100 的 pass@1 为 `24.83% / 25.67% / 33.83% / 31.50% / 38.17% / 38.00%`，模型跨度 `13.33pp`；把六个模型固定后只改变 test prompt，C0/C1/C2 总体为 `33.17% / 31.67% / 31.17%`，跨度仍只有 `2.00pp`。
2. **shared SFT 单独几乎没有提高 held-out pass@1。** SFT-only 相对 raw Base 仅 `+0.83pp [−2.33, 4.17]`；三个 test prompt 的 cell 差异也只有 `+0.5/+1.0/+1.0pp`，exact McNemar `p≥0.83`。因此原来 Base→trained 的大部分收益不能再笼统归给共同 SFT。
3. **主要收益来自后续 GRPO，而且策略分支差异是真实的。** 相对同一个 SFT-only 初态，selected C0/C1/C2 分别提高 `+8.17pp [4.67, 12.00]`、`+12.50pp [8.67, 16.67]`、`+12.33pp [8.33, 16.33]`。在统一 step 100 后，C1−C0-step100 为 `+6.67pp [3.67, 9.83]`，C2−C0-step100 为 `+6.50pp [2.50, 10.50]`；这比 selected-checkpoint 的约 4pp 差异更直接地排除了训练步数混杂。
4. **C0 的 step 100 没有优于被选中的 step 40。** C0-step100 平均 `31.50%`，比 C0-step40 低 `−2.33pp [−5.50, 0.83]`；三种 prompt 下均为负但均不显著。它说明 C0 后 60 步没有带来可迁移收益，并伴随格式合法率从 `84.00%` 降到 `73.33%`、cap-hit 从 `16.67%` 升到 `27.00%`。
5. **更详细的 test prompt 仍没有稳定收益。** 原 4 个模型的 12 组模型内比较均不显著；新增 C0-step100 与 SFT-only 的 6 组比较在 Holm 校正后也全部不显著。三个 selected trained model 上，test C1 相对 C0 为 `−2.17pp [−5.17, 0.83]`，test C2 相对 C0 为 `−2.83pp [−6.33, 0.50]`。
6. **没有可靠的 train/test 同名匹配收益。** 三条 selected GRPO 分支的对角线均值为 `37.33%`，六条非对角线均值为 `36.33%`，匹配溢价只有 `+1.00pp [−1.08, 3.17]`。
7. **C0 学到的是“可靠完成协议”，C1/C2 学到的是“更高质量的已提交代码”。** selected C0 的格式合法率最高（`84.00%`）、cap-hit 最低（`16.67%`），但合法提交后的通过率只有 `40.28%`；C1/C2 分别为 `58.72% / 54.55%`。C0-step100 的条件通过率虽升到 `42.95%`，却因终止退化而降低总 pass。
8. **cap-hit 仍是最大的失败机制。** 合并 3,600 条 evaluation 轨迹后有 1,080 次 cap-hit、1,051 次格式失败；**全部格式失败都伴随 cap-hit**，cap-hit 条件下 `0/1,080` 通过，未 cap 时通过率 `45.71%`。新增数据没有改变原机制判断。
9. **C1 与 C2 正确率相当，但 C2 更省。** C1/C2 pass@1 只差 `0.17pp`；C2 平均比 C1 少 `745` completion tokens、少 `751` trajectory tokens、快 `9.15s`，同时 format-valid 高 `4.67pp`、cap-hit 低 `5.33pp`。
10. **结论仍受 post-hoc、单 seed 与复用测试集限制。** C0-step100 和 SFT-only 是看过原 final 后补做的机制消融，不能升级为新的 confirmatory test；它们强化了归因，但正式推广仍需 fresh split 或多 seed 复验。

## 1. 实验与因果口径

### 1.1 三种 system prompt

三种 prompt 共用任务、工具协议、输出格式与最大一次 `run_python` 调用；只改变附加策略说明。

| Prompt | 额外策略 | 预期作用 |
|---|---|---|
| C0 | 无 | 只保证任务、工具、最终代码格式协议 |
| C1 | 简洁分析 I/O、约束、算法和边界；只做一次聚焦检查；尽快提交 | 降低无效长推理，让工具使用更有目的 |
| C2 | 六步工作流：理解 → 算法/复杂度 → 实现 → 样例检查 → 边界修复 → 最终复核 | 强化系统化解题与自检 |

### 1.2 哪些比较能说明什么

| 比较 | 可解释为 | 主要限制 |
|---|---|---|
| 同一模型、同一道题，C0/C1/C2 test prompt 对比 | 干净的 test-time prompt 效应 | greedy 远程推理仍可能有少量服务端非确定性 |
| SFT-only vs raw Base | 共同 neutral SFT 的边际贡献 | post-hoc 且复用原 test set；只有一个 SFT seed |
| C0/C1/C2 vs SFT-only | 各 GRPO 分支相对共同 SFT 初态的增益 | selected checkpoint 不同；只有一个训练 seed |
| C1/C2-step100 vs C0-step100 | 同训练步数下的 GRPO prompt/策略分支差异 | post-hoc；独立分支仍有优化随机性 |
| C0-step100 vs C0-step40 | C0 分支 checkpoint-step 敏感性 | step40 是 dev-selected；比较为 post-hoc |
| 任一 trained model vs raw Base | 完整 `shared SFT + GRPO` pipeline 的收益 | **不能**把全部收益归因于 system prompt 或 GRPO |
| 对角线 vs 非对角线 | train/test prompt matching 或 overfit | 只有三个 prompt、一个 seed，功效有限 |

### 1.3 数据规模与运行健康度

| 阶段 | 规模 | 完整性 |
|---|---:|---|
| GRPO train | 3 branches × 100 steps × 4 questions × group 8 = 9,600 trajectories | 300/300 step metrics；正式目录每 step 恰好 32 条轨迹 |
| checkpoint-dev | 3 branches × 5 checkpoints × 3 prompts × 100 tasks = 4,500 | 4,500/4,500 completed，0 failed |
| selected-checkpoint final | 4 models × 3 prompts × 200 tasks = 2,400 | 2,400/2,400 completed，0 failed |
| post-hoc final | 2 models × 3 prompts × 200 tasks = 1,200 | 1,200/1,200 completed，0 failed |

训练与两批 final 的 Docker infrastructure error 均为 0。selected final 三个 prompt shard 各 800/800 completed，post-hoc 三个 shard 各 400/400 completed。统计结果不是由失败任务补零或缺失 cell 造成。

## 2. Selected-checkpoint Final 主指标：12-cell Train × Test 矩阵

### 2.1 Pass@1

| Train model \ Test prompt | C0 | C1 | C2 | Average | Worst | OverfitGap |
|---|---:|---:|---:|---:|---:|---:|
| C0 (step 40) | 36.0% | 34.0% | 31.5% | 33.83% | 31.5% | +3.17pp |
| C1 (step 100) | 38.5% | 38.5% | 37.5% | **38.17%** | **37.5%** | +0.50pp |
| C2 (step 100) | **40.5%** | 36.0% | 37.5% | 38.00% | 36.0% | −0.83pp |
| Base | 25.0% | 24.0% | 25.5% | 24.83% | 24.0% | — |

解释：

- 单格最高是 `C2 train × C0 test = 40.5%`，但三 prompt 平均最高是 C1 的 `38.17%`。
- C1 的三个 cell 只相差 1pp，Worst 也是最高，说明它对 test prompt 最稳健。
- C2 对 test C0 最强，但到 test C1 会下降 4.5pp；这个 cell 内差异在配对检验中没有显著，不能解释成确定的 prompt 冲突。
- C0 随 test prompt 变复杂呈 `36.0 → 34.0 → 31.5%`，但三组 pairwise CI 均跨 0；方向值得关注，证据尚不足以宣称确定伤害。

### 2.2 Trained model 相对 Base：配对效应

| Train | Test | Δ pass@1 | 95% paired bootstrap CI | Wins / Ties / Losses | exact McNemar p | Holm p |
|---|---|---:|---|---:|---:|---:|
| C0 | C0 | +11.0pp | [5.5, 17.0] | 30 / 162 / 8 | 0.00047 | 0.00155 |
| C0 | C1 | +10.0pp | [4.0, 16.0] | 31 / 158 / 11 | 0.00289 | 0.00577 |
| C0 | C2 | +6.0pp | [0.5, 11.5] | 21 / 170 / 9 | 0.04277 | 0.04277 |
| C1 | C0 | +13.5pp | [7.5, 19.5] | 34 / 159 / 7 | 0.000025 | 0.000152 |
| C1 | C1 | +14.5pp | [8.5, 20.5] | 35 / 159 / 6 | 0.0000049 | 0.000039 |
| C1 | C2 | +12.0pp | [6.5, 17.5] | 29 / 166 / 5 | 0.000039 | 0.000193 |
| C2 | C0 | **+15.5pp** | [10.0, 21.0] | 34 / 163 / 3 | 0.00000012 | 0.0000011 |
| C2 | C1 | +12.0pp | [7.0, 17.5] | 28 / 168 / 4 | 0.000019 | 0.000135 |
| C2 | C2 | +12.0pp | [6.0, 18.5] | 34 / 156 / 10 | 0.000388 | 0.00155 |

九个 cell 在 9 次 Holm 校正后仍达到 0.05。完整训练 pipeline 的收益不是少数样本偶然翻转：每个 cell 都有明显更多的 trained-only win than Base-only loss。

但要再次强调：这里的 Base 是 raw `Qwen3.5-4B`，三条 trained branch 先经过共同的 shared-neutral SFT，再经过各自 GRPO；因此该表证明的是完整训练流程有效，不是“某段 system prompt 单独带来 6–15.5pp”。

### 2.3 新增 post-hoc final：C0-step100 与 SFT-only

| Model \ Test prompt | C0 | C1 | C2 | Average | Worst |
|---|---:|---:|---:|---:|---:|
| C0-step100 | 33.5% | 32.5% | 28.5% | 31.50% | 28.5% |
| shared-SFT-only | 25.5% | 25.0% | 26.5% | 25.67% | 25.0% |

两个新增模型都在与原 final 相同的 200 道题和三个 prompt 上完成。C0-step100 的三个 cell 都低于 selected C0-step40；SFT-only 则在三个 prompt 上都与 raw Base 基本重合。

| Post-hoc 对比 | Test | Δ pass@1 | 95% paired CI | Wins / Ties / Losses | exact McNemar p |
|---|---|---:|---|---:|---:|
| C0-step100 − C0-step40 | C0 | −2.5pp | [−7.5, 2.5] | 10 / 175 / 15 | 0.424 |
| C0-step100 − C0-step40 | C1 | −1.5pp | [−6.5, 4.0] | 13 / 171 / 16 | 0.711 |
| C0-step100 − C0-step40 | C2 | −3.0pp | [−8.5, 2.0] | 11 / 172 / 17 | 0.345 |
| SFT-only − Base | C0 | +0.5pp | [−4.5, 5.5] | 14 / 173 / 13 | 1.000 |
| SFT-only − Base | C1 | +1.0pp | [−3.5, 5.5] | 12 / 178 / 10 | 0.832 |
| SFT-only − Base | C2 | +1.0pp | [−4.0, 6.5] | 15 / 172 / 13 | 0.851 |

这些是复用原 test set 的 post-hoc 对比，因此以效应量和区间为主，不应把“未显著”解释成严格等价。

## 3. 分离 train prompt 与 test prompt 的影响

### 3.1 Train prompt 的边际效应

每道题先在三个 test prompt 上取均值，再在 200 题上做 paired bootstrap，避免把同一道题的三个重复条件当作 600 个独立样本。

| 对比 | Δ pass@1 | 95% CI | 结论 |
|---|---:|---|---|
| SFT-only − Base | +0.83pp | [−2.33, 4.17] | neutral SFT 单独没有可分辨的 pass@1 增益 |
| C0-step40 − SFT-only | +8.17pp | [4.67, 12.00] | selected C0 的主要收益来自后续 GRPO |
| C0-step100 − SFT-only | +5.83pp | [2.33, 9.50] | C0 GRPO 到 step100 仍优于 SFT-only |
| C1-step100 − SFT-only | +12.50pp | [8.67, 16.67] | C1 GRPO 带来最大的边际提升 |
| C2-step100 − SFT-only | +12.33pp | [8.33, 16.33] | C2 GRPO 与 C1 几乎相同 |
| C0-step100 − C0-step40 | −2.33pp | [−5.50, 0.83] | C0 后续训练未提高 held-out final |
| C1-step100 − C0-step100 | **+6.67pp** | [3.67, 9.83] | 同步数下简洁策略优于纯协议 |
| C2-step100 − C0-step100 | **+6.50pp** | [2.50, 10.50] | 同步数下结构化策略优于纯协议 |
| C2 − C1 | −0.17pp | [−3.83, 3.33] | 六步工作流没有比简洁策略更高的 pass@1 |

新增对照把原来的 `raw Base → shared SFT → prompt-conditioned GRPO` 拆开了：**shared SFT 单独几乎不改变 held-out pass@1，主要增益在 GRPO 阶段；而同为 step 100 时 C1/C2 仍稳定领先 C0，说明“有策略说明”的优势不能用 C0 checkpoint 选得更早来解释。** 但 post-hoc 复用测试集，结论强度仍低于预注册 fresh test。

### 3.2 Test prompt 的边际效应

在三个 trained model 内做同题配对：

| Test prompt 对比 | Δ pass@1 | 95% CI |
|---|---:|---|
| C1 − C0 | −2.17pp | [−5.17, 0.83] |
| C2 − C0 | −2.83pp | [−6.33, 0.50] |
| C2 − C1 | −0.67pp | [−3.67, 2.33] |

对应的 secondary metric 也没有系统性变化：三个 trained model 的 C0/C1/C2 test prompt 格式合法率为 `72.50% / 72.83% / 73.33%`，cap-hit 为 `28.33% / 27.83% / 26.83%`，case-pass 为 `48.58% / 46.82% / 47.29%`。这些差异都很小，配对 CI 均跨 0。

结论是：**详细 test prompt 没有把训练中缺失的能力临时“提示出来”。** 最终模型行为主要由权重分支决定，test prompt 只会在个别题上改变成功与否，并没有一致方向。

新增两组也一致：C0-step100 的 C1−C0 / C2−C0 / C2−C1 分别为 `−1.0pp [−6.0, 4.0]`、`−5.0pp [−10.5, 0.01]`、`−4.0pp [−8.5, 0.5]`；SFT-only 分别为 `−0.5pp`、`+1.0pp`、`+1.5pp`，区间均跨 0。合并六个模型后，test C0/C1/C2 的总体 pass@1 为 `33.17% / 31.67% / 31.17%`，仍未出现“prompt 越详细越好”。

### 3.3 Train/Test matching

| 统计量 | 值 |
|---|---:|
| 对角线（C0×C0、C1×C1、C2×C2）均值 | 37.33% |
| 非对角线六 cell 均值 | 36.33% |
| Matching premium | +1.00pp |
| 95% paired bootstrap CI | [−1.08, 3.17] |

OverfitGap 也支持同一结论：C0 `+3.17pp`、C1 `+0.50pp`、C2 `−0.83pp`。只有 C0 有较明显的“匹配 prompt 更好”点估计，但仍不足以证明系统性 overfit。C1/C2 的策略更像被内化为权重行为，而不是依赖测试时复现同一句 prompt。

## 4. Secondary metrics：正确率之外发生了什么

### 4.1 能力、格式与错误

下表跨三个 test prompt 聚合，每个模型 600 条轨迹；`C0@40` 是原 selected checkpoint，`C0@100` 与 SFT-only 是新增 post-hoc。

| Metric | Base | SFT-only | C0@40 | C0@100 | C1@100 | C2@100 |
|---|---:|---:|---:|---:|---:|---:|
| pass@1 | 24.83% | 25.67% | 33.83% | 31.50% | **38.17%** | 38.00% |
| case-pass | 35.36% | 37.80% | **48.05%** | 43.35% | 47.65% | 46.99% |
| public-pass | 39.73% | 42.78% | **54.79%** | 49.53% | 52.12% | 49.93% |
| private-pass | 35.06% | 37.42% | **47.21%** | 42.74% | 47.08% | 46.60% |
| format-valid | 63.33% | 69.50% | **84.00%** | 73.33% | 65.00% | 69.67% |
| token cap-hit | 37.67% | 32.33% | **16.67%** | 27.00% | 35.83% | 30.50% |
| compile error | 0.33% | 0.17% | **0.00%** | 0.33% | 0.33% | 0.67% |
| runtime error | 8.17% | 8.83% | 11.50% | 8.67% | **4.83%** | 8.83% |
| time limit | 2.83% | 4.50% | 4.00% | 4.00% | 2.83% | **1.33%** |
| Pass given valid format | 39.21% | 36.93% | 40.28% | 42.95% | **58.72%** | 54.55% |

主要解释：

- **三条 trained branch 的 case-pass 几乎相同，但 pass@1 不同。** C0 case-pass 甚至最高，说明它经常通过大部分 case，却更容易在边界条件上留下最后一两个错误；C1/C2 更常把“部分正确”推到“全测试通过”。
- **C0 的 public/private gap 最大。** C0 为 `54.79−47.21=7.58pp`，Base/C1/C2 分别为 `4.67/5.04/3.33pp`。这可能意味着 C0 更偏向可见样例或 public case，也可能只是 public case 更容易；现有数据不能把两者分开。
- **C1 的已提交代码质量最好。** C1 虽然格式合法率只有 65%，一旦合法提交，通过率为 58.72%；C0 只有 40.28%。因此 C1 的主要瓶颈不是算法质量，而是长生成没有及时形成合法最终提交。
- **C0 的代价是运行错误和超时。** 它格式最稳定，却有最多 runtime error（69/600）和 time limit（24/600），与“更快提交、较少完整自检”的行为一致。
- **SFT-only 主要改善了协议外观而不是解题正确性。** 相对 Base，format-valid `+6.17pp`、cap-hit `−5.33pp`，但合法提交后的通过率反而低 `2.28pp`，最终 pass 仅 `+0.83pp`。
- **C0 后 60 步发生了终止退化。** C0@100 相对 C0@40 的 case-pass、format-valid 和总 pass 分别低 `4.70/10.67/2.33pp`，cap-hit 高 `10.33pp`；虽然合法提交后的通过率高 `2.68pp`，不足以抵消提交完成率下降。
- compile error 全部很低，不是主要瓶颈。

### 4.2 完整失败类型分布

这里的 `Invalid format` 是 **judge 的最终状态**：模型最终回复没有被解析成“且仅有一个非空的 Python fenced code block”，因此没有可送去执行判题的 `final_code`。它本身不等同于“被截断”；多余正文、空代码块、围栏不完整、非法/未完成 tool call 或始终没有 final submission 都可能触发。截断是另一项轨迹标记 `hit_token_limit`（汇总为 `token_cap_hit_rate`）。不过在本报告的 L10K/20K 3,600 条数据中，两者高度共现：1,051 条 invalid-format 全部 hit cap；另有 29 条 hit cap 最终落入其他失败状态。因此可说这里的 invalid-format **主要由截断造成**，但不能把两个指标当作定义上的同义词。

| Final status | Base | SFT-only | C0@40 | C0@100 | C1@100 | C2@100 |
|---|---:|---:|---:|---:|---:|---:|
| Pass | 149 (24.8%) | 154 (25.7%) | 203 (33.8%) | 189 (31.5%) | **229 (38.2%)** | 228 (38.0%) |
| Invalid format | 220 (36.7%) | 183 (30.5%) | **96 (16.0%)** | 160 (26.7%) | 210 (35.0%) | 182 (30.3%) |
| Wrong answer | 163 (27.2%) | 182 (30.3%) | 208 (34.7%) | 173 (28.8%) | **113 (18.8%)** | 125 (20.8%) |
| Runtime error | 49 (8.2%) | 53 (8.8%) | 69 (11.5%) | 52 (8.7%) | **29 (4.8%)** | 53 (8.8%) |
| Time limit | 17 (2.8%) | 27 (4.5%) | 24 (4.0%) | 24 (4.0%) | 17 (2.8%) | **8 (1.3%)** |
| Compile error | 2 (0.3%) | 1 (0.2%) | 0 | 2 (0.3%) | 2 (0.3%) | 4 (0.7%) |

这张表展示了明显的 failure-mode exchange：

- C0 把大量 Base 的 invalid-format 修成了可执行程序，但相当一部分落在 wrong-answer/runtime/time-limit。
- C1/C2 一旦提交，更少落入 wrong-answer；但它们仍有 30–35% 的 invalid-format。
- SFT-only 把 37 个 invalid-format 转移到了其他状态，但净增只有 5 个 pass；C0@100 又进一步减少 23 个 invalid-format、净增 35 个 pass，仍明显弱于 C1/C2。
- 因此后续优化不应只继续提高 reward/pass；对 C1/C2 需要单独解决终止与 final submission，对 C0 则要加强边界和复杂度检查。

#### 未通过题目中的截断占比：L10K/20K vs L16K/24K

口径：`未通过 = pass@1=0`；`被截断 = 任一 assistant turn 的 hit_token_limit=true`。表中分母是每个 cell 最终未通过的题数，不是该 cell 的全部 200 题。两种长度条件下均不存在“hit cap 但最终 pass”的轨迹，因此分子也等于该 cell 的 cap-hit 总数。L16K/24K 只重跑了 Base 与三个 selected checkpoint，故这里不列 SFT-only 和 C0@100。

| Train model | Test prompt | L10K/20K：截断 / 未通过 | L16K/24K：截断 / 未通过 |
|---|---|---:|---:|
| Base | C0 | 70/150 (46.67%) | 78/154 (50.65%) |
| Base | C1 | 77/152 (50.66%) | 68/154 (44.16%) |
| Base | C2 | 79/149 (53.02%) | 71/150 (47.33%) |
| C0@40 | C0 | 31/128 (24.22%) | 36/132 (27.27%) |
| C0@40 | C1 | 37/132 (28.03%) | 30/137 (21.90%) |
| C0@40 | C2 | 32/137 (23.36%) | 36/133 (27.07%) |
| C1@100 | C0 | 79/123 (64.23%) | 69/127 (54.33%) |
| C1@100 | C1 | 68/123 (55.28%) | 60/118 (50.85%) |
| C1@100 | C2 | 68/125 (54.40%) | 77/125 (61.60%) |
| C2@100 | C0 | 60/119 (50.42%) | 51/119 (42.86%) |
| C2@100 | C1 | 62/128 (48.44%) | 48/121 (39.67%) |
| C2@100 | C2 | 61/125 (48.80%) | 47/123 (38.21%) |
| **总体** | — | **724/1,591 (45.51%)** | **671/1,593 (42.12%)** |

这比单看 cap-hit / 全部题更贴近“失败是否由长度终止主导”：L10K/20K 下，C1@100 三个 cell 的失败中有 `54.40%–64.23%` 被截断，而 C0@40 只有 `23.36%–28.03%`；L16K/24K 后总体占比从 `45.51%` 降至 `42.12%`，但并非所有 cell 都下降，Base×C0、C0@40×C0/C2 与 C1@100×C2 反而上升。

### 4.3 Cap-hit 与格式失败

| 统计 | 值 |
|---|---:|
| Evaluation trajectories | 3,600 |
| Cap-hit | 1,080 (30.00%) |
| Invalid format | 1,051 (29.19%) |
| Cap-hit ∩ invalid format | 1,051 |
| P(invalid format \| cap-hit) | 97.31% |
| P(cap-hit \| invalid format) | **100.00%** |
| Pass rate under cap-hit | **0.00%** |
| Pass rate without cap-hit | 45.71% |

这里不能简单把 cap-hit 当成外部长度不足：最大 assistant 长度已经从原 mini 的 2K 扩到 10,240。更准确的解释是，部分策略分支仍会进入过长、未收束的生成模式。新增 1,200 条数据后，“所有 invalid-format 都来自 cap-hit、cap-hit 下无一通过”的关系完全保持，说明它既是 inference budget 问题，也是训练出的行为问题。

### 4.4 工具行为

| Metric | Base | SFT-only | C0@40 | C0@100 | C1@100 | C2@100 |
|---|---:|---:|---:|---:|---:|---:|
| 尝试调用工具的轨迹比例 | 92.50% | 96.67% | 99.83% | 99.67% | 100.00% | 100.00% |
| 有效调用 / 尝试调用 | 63.96% | 68.45% | **84.14%** | 73.58% | 65.00% | 69.67% |
| 至少一次有效工具调用 | 59.17% | 66.17% | **84.00%** | 73.33% | 65.00% | 69.67% |
| Mean tool calls | 0.592 | 0.662 | **0.840** | 0.733 | 0.650 | 0.697 |
| Mean turns | 1.592 | 1.662 | **1.840** | 1.733 | 1.650 | 1.697 |
| Pass given valid tool use | 38.31% | 36.02% | 40.28% | 42.95% | **58.72%** | 54.55% |
| Pass with no valid tool call | 5.31% | 5.42% | 0% | 0% | 0% | 0% |

由于协议最多一次工具调用，`mean_tool_calls`、单轨迹 `tool_call_valid_rate` 与有效工具使用比例数值接近；真正能看出解析质量的是“有效调用/尝试调用”。

行为含义：

- 三个 trained model 几乎每题都会尝试工具，C1 中“只在有具体不确定性时使用”的文字没有形成可见的选择性工具使用。
- C1/C2 没有有效工具调用时 0 条通过。这里的“无有效调用”多数是非法/未完成 tool call，而不是一个干净的 no-tool control，不能据此声称工具具有因果收益。
- C0 最大的优势之一是工具协议完成率；C1/C2 则在工具路径成功后产生质量更高的最终程序。
- SFT-only 只把有效调用率从 Base 的 `59.17%` 提到 `66.17%`；C0 GRPO 才进一步把协议完成率推高。C0@100 相对 C0@40 的有效调用率回落 `10.67pp`，与其终止退化一致。
- 后续可加入 `tool-needed` 或“直接提交 vs 工具检查”的显式行为奖励，避免模型把工具调用学成强制仪式。

### 4.5 Token、时延与执行成本

| Metric | Base | SFT-only | C0@40 | C0@100 | C1@100 | C2@100 |
|---|---:|---:|---:|---:|---:|---:|
| Mean prompt tokens | 1,166.7 | 1,166.7 | 1,166.7 | 1,166.7 | 1,166.7 | 1,166.7 |
| Mean completion tokens | 4,563 | 4,195 | **3,643** | 4,542 | 5,624 | 4,880 |
| Mean trajectory tokens | 5,812 | 5,451 | **4,950** | 5,842 | 6,889 | 6,137 |
| Mean latency | 41.3s | 78.6s | 57.5s | 86.4s | 86.9s | 77.7s |
| Mean judge execution | 0.750s | 0.753s | 0.947s | 0.777s | 0.684s | **0.604s** |

配对效应：

- C0 vs Base：completion `−920 [−1288, −546]`，但 latency `+16.18s [11.71, 20.75]`。C0 输出更短，却因为更多有效工具回合而更慢。
- C1 vs C0：pass@1 `+4.33pp`，代价是 completion `+1,982`、trajectory `+1,939`、latency `+29.43s`。
- C2 vs C0：pass@1 `+4.17pp`，代价是 completion `+1,237`、trajectory `+1,188`、latency `+20.28s`。
- C2 vs C1：pass@1 无差异，但 completion `−745 [−1040, −452]`、trajectory `−751`、latency `−9.15s [−13.82, −4.40]`。
- C0@100 vs C0@40：completion `+899`、trajectory `+892`，但 pass@1 `−2.33pp`；C0 后续训练把更多预算花在更长、较难收束的生成上，没有换来正确率。

这使 C2 在 C1/C2 之间形成更好的 Pareto 点：正确率相当、成本更低。C0 仍是最省 token 的分支，但不是最低 wall-clock，因为它更常完成工具回合和 judge 执行。

SFT-only/C0@100 与原四个模型不在同一批远程请求中，latency 容易受服务负载影响；因此上表的跨批 wall-clock 只作描述，不用于因果排序。token 与 pass 的同题配对比较不受这项运行时漂移影响。

### 4.6 Test prompt 自身的成本

在六个模型上，C0/C1/C2 的平均 prompt tokens 仍为 `1108.4 / 1176.4 / 1215.4`，而 pass@1 为 `33.17% / 31.67% / 31.17%`。C2 只比 C0 多 107 prompt tokens，却没有带来正确率收益；selected trained model 的 completion/trajectory/latency 在三种 test prompt 间差异也不显著。因此高成本主要来自权重学出的生成行为，而不是 test prompt 多出的百余 token。

## 5. 训练过程分析

训练时 temperature=1、每 step 4 题 × group 8；单 step 波动很大，不能把训练 `reward/pass_at_1` 当作泛化结果。以下主要看 100-step 平均、20-step 窗口与 checkpoint-dev。

### 5.1 100-step 总体行为

| Metric | C0 | C1 | C2 |
|---|---:|---:|---:|
| reward/mean | 67.22% | **69.45%** | 66.14% |
| reward/pass_at_1 | 53.34% | **57.84%** | 56.81% |
| reward/case_pass_rate | 67.62% | **70.14%** | 67.15% |
| reward/public_pass_rate | 70.45% | **74.59%** | 70.37% |
| reward/private_pass_rate | 66.94% | **69.17%** | 66.42% |
| rollout/format_valid_rate | **96.00%** | 93.06% | 89.81% |
| rollout/token_cap_hit_rate | **3.69%** | 6.50% | 9.81% |
| rollout/tool_use_rate | **95.87%** | 91.16% | 88.94% |
| rollout/valid_tool_call_rate | **96.03%** | 93.30% | 89.86% |
| rollout/mean_turns | 1.959 | 1.912 | 1.889 |
| rollout/mean_tokens | **3,565** | 3,938 | 4,238 |
| rollout/degenerate_group_rate | **29.25%** | 38.25% | 31.00% |
| train/update_skipped | 1.00% | 6.00% | **0.00%** |
| train/datums_per_step | 22.47 | 19.56 | 22.02 |
| train/micro_batches_per_step | 2.58 | 2.55 | 2.90 |
| train/overlength_datum_rate | 0.146% | **0.031%** | 0.083% |
| trainer/loss_mean | 8.06e−4 | **3.26e−4** | 6.46e−4 |
| trainer/token_count | 23,684 | 25,323 | 25,712 |
| docker/infrastructure_error_rate | 0 | 0 | 0 |
| docker/timeout_rate | 2.74% | 1.99% | **1.93%** |
| docker/mean_seconds | 1.93s | 1.95s | **1.83s** |
| time/step_seconds | **185.0s** | 204.1s | 225.8s |

`importance_sampling` loss 在 0 附近正负变化，不应像 SFT cross-entropy 一样按“越低越好”排序；这里 loss 只用于检查数值稳定性。三支都没有 infrastructure error，overlength datum 也低于 0.15%，说明 final 差异不是训练执行器失稳导致。

### 5.2 训练前 20 步到后 20 步的变化

| Branch | Δ reward mean | Δ train pass | Δ format-valid | Δ cap-hit | Δ mean tokens | Δ degenerate group | Last-20 skipped update |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | +5.38pp | +5.62pp | **+2.82pp** | **−2.66pp** | +223 | +11.25pp | 5% |
| C1 | +10.08pp | +17.81pp | −4.69pp | +5.78pp | +1,399 | +21.25pp | 5% |
| C2 | **+11.48pp** | **+18.13pp** | −5.16pp | +5.47pp | +671 | +25.00pp | 0% |

C1/C2 学到更高 reward/pass 的同时，生成变长、cap-hit 上升、格式合法下降；C0 则朝“更短、更稳定完成”的方向发展。这一训练期行为与 final failure mode 完全一致，说明不是 final 偶然噪声。

不过训练题按固定顺序每题只使用一次，前 20 与后 20 的题目难度可能不同；因此上表不能单独作为学习曲线因果证据，必须与固定 100 题 checkpoint-dev 一起解释。

## 6. Checkpoint-dev：能力如何随 step 变化

每个 checkpoint 都在同一 100 题 × 三 prompt 上评测，下面的数值是跨 prompt 平均。

| Branch | Step | pass@1 | case-pass | format-valid | cap-hit | tool-use | trajectory tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 20 | 55.00% | 62.41% | 88.33% | 12.00% | 88.33% | 3,781 |
| C0 | **40** | **56.67%** | 64.05% | **94.33%** | **6.00%** | **94.33%** | **3,379** |
| C0 | 60 | 51.67% | 61.46% | 90.00% | 10.00% | 90.00% | 3,740 |
| C0 | 80 | **56.67%** | **65.95%** | 93.00% | 7.00% | 93.00% | 3,514 |
| C0 | 100 | 56.00% | 63.78% | 90.33% | 9.67% | 90.33% | 4,068 |
| C1 | 20 | 48.33% | 57.61% | 85.67% | 14.67% | 85.67% | **3,529** |
| C1 | 40 | 56.33% | 65.19% | **87.33%** | **13.00%** | **87.33%** | 4,001 |
| C1 | 60 | 61.33% | 67.75% | 86.00% | 14.33% | 86.00% | 4,411 |
| C1 | 80 | 60.00% | 66.50% | 82.33% | 17.67% | 82.33% | 4,901 |
| C1 | **100** | **62.00%** | **68.40%** | **87.33%** | **13.00%** | **87.33%** | 4,568 |
| C2 | 20 | 53.00% | 60.89% | 79.67% | 20.33% | 77.33% | 4,129 |
| C2 | 40 | 49.67% | 56.61% | 74.00% | 26.00% | 73.67% | 4,799 |
| C2 | 60 | 55.33% | 61.99% | 82.33% | 18.00% | 82.33% | 4,133 |
| C2 | 80 | 56.00% | 62.95% | 79.67% | 20.33% | 79.67% | 4,420 |
| C2 | **100** | **64.33%** | **71.76%** | **90.67%** | **9.67%** | **90.33%** | **3,784** |

结论：

- C0 很早达到平台，step 20–100 没有单调能力增长；step 40/80 pass 相同，step 40 因 Worst 更高被选中。
- C1 整体随训练上升，step 100 最佳；中后期 trajectory 变长并伴随格式波动。
- C2 是明显的“晚熟”分支：step 40 一度退化，step 100 同时出现 pass、case-pass、format 和 cap 的全面改善。
- checkpoint-dev 在固定 step 100 上是 C2/C1 领先 C0 `8.33/6.00pp`；新增 final 又给出 C2/C1 领先 C0-step100 `6.50/6.67pp`。dev 与 held-out final 的方向一致，因此原 final 差异不能归因于 C0 只使用 step 40。

### 6.1 Dev → final 的落差

| Branch/checkpoint | Dev average | Final average | Drop |
|---|---:|---:|---:|
| C0-step40 (selected) | 56.67% | 33.83% | −22.84pp |
| C0-step100 (post-hoc) | 56.00% | 31.50% | −24.50pp |
| C1-step100 (selected) | 62.00% | 38.17% | −23.83pp |
| C2-step100 (selected) | 64.33% | 38.00% | **−26.33pp** |

四个 checkpoint 都出现 23–26pp 的下降，说明 checkpoint-dev 与 temporally held-out test 存在明显难度/分布差异，也包含“从 5 个 checkpoint 中选最好者”的选择乐观性。C0-step100 在 dev 只比 step40 低 `0.67pp`，到 final 却低 `2.33pp`，进一步说明 100 题 dev 上的微小排序不能可靠外推。

## 7. 难度与平台分层

### 7.1 Difficulty

下表仍跨三个 test prompt 聚合；`n` 是唯一题目数。

| Difficulty | n | Base | SFT-only | C0@40 | C0@100 | C1@100 | C2@100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Easy | 46 | 65.22% | 67.39% | 84.06% | 85.51% | **94.93%** | 92.75% |
| Medium | 67 | 25.37% | 23.38% | 30.85% | 25.87% | 35.32% | **38.81%** |
| Hard | 87 | 3.07% | 5.36% | 9.58% | 7.28% | **10.34%** | 8.43% |

训练收益在 easy/medium 最明显；hard 题虽然相对 Base 提高约 5–7pp，但绝对通过率仍只有 8–10%。这说明当前流程首先修复了协议、已知模板和中等复杂度边界处理，尚未显著解决真正的高难算法推导。

SFT-only 在 easy/hard 上略高、medium 上略低，仍与 Base 基本同层；它没有解释后续 GRPO 的大幅提升。C0@100 的 easy 略高于 C0@40，但 medium/hard 分别低 `4.98/2.30pp`。hard 题 cap-hit 从 C0@40 的 `32.18%` 回升到 C0@100 的 `43.68%`，也解释了后期训练为何未能提升 overall final。

### 7.2 Platform

| Platform | n | Base | SFT-only | C0@40 | C0@100 | C1@100 | C2@100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| AtCoder | 122 | 26.78% | 27.60% | 33.06% | 30.05% | **34.15%** | 33.61% |
| LeetCode | 78 | 21.79% | 22.65% | 35.04% | 33.76% | 44.44% | **44.87%** |

SFT-only 在两个平台上都只比 Base 高约 `0.8pp`；GRPO 分支的提升才明显扩大，且 C1/C2 对 LeetCode 的提升大于 AtCoder。平台与 difficulty、starter-code/function-style、题型分布互相混杂，这里仍只能做描述性结论。

## 8. 单题下钻：聚合数字如何落到具体行为

### 8.1 训练修复终止失败

`3605 / construct-the-minimum-bitwise-array-i`（LeetCode easy）：Base 在 C0/C1/C2 下全部 cap-hit + invalid-format，三个 selected GRPO model 的九个 cell 全部通过。数据中共有 6 道题满足“Base 三 prompt 全失败、所有 9 个 selected GRPO cell 全通过”，而且都是 easy。结合 SFT-only 总体仅比 Base 高 `0.83pp`，这类全面修复更可能主要发生在后续 GRPO 阶段；单题层面仍不把聚合归因强行外推为因果结论。

### 8.2 Test prompt 在个别题上能改变结果，但方向不一致

`3603 / check-if-dfs-strings-are-palindromes`（LeetCode hard）：

- C1 model：C0 test 为 wrong-answer/cap，C1 test 通过，C2 test compile-error；
- C2 model：C0 test 通过，C1 runtime-error，C2 wrong-answer。

这说明同一道题确实可能被 prompt 改写触发不同推理路径，但“更详细”并不保证更好，也没有形成总体可复现方向。数据中有 20 道题是 C1 model 在 C0 test 失败、但 C1 或 C2 至少一个通过；同时有 8 道题是 C2 model 只有 C0 test 通过、C1/C2 都失败。

### 8.3 部分通过率很高仍可能因一个边界失败

`3496 / minimum-number-of-seconds-to-make-mountain-height-zero`（LeetCode medium）：C0/C1 model 在 C0 test 都达到 `97.67%` case-pass 但不是全通过；C1/C2 test 能使部分 cell 变成 100%。这类样本解释了为什么 C0 的 case-pass 最高、pass@1 却较低：不是完全不会，而是最后的边界或数值细节没有处理完。

### 8.4 长生成仍是跨模型共同难题

共有 44 道题满足“Base/C0/C1/C2 每个模型至少在一种 test prompt 下发生 cap-hit”。例如 `3510 / maximize-the-total-height-of-unique-towers`，12 个 cell 中 6 个 cap-hit，且无一通过。此类题是后续 termination/conciseness 训练最有价值的定向样本池。

## 9. 回答研究问题：train 与 test system prompt 究竟产生了什么影响

### 9.1 Train system prompt

新增 SFT-only 先排除了一个重要替代解释：neutral SFT 只把 pass@1 从 `24.83%` 推到 `25.67%`，而后续 GRPO 才把它推到 `31.50–38.17%`。Train prompt 改变的不是表面输出措辞，而是三种稳定的权重行为：

- **C0：协议完成型。**工具调用最有效、输出最短、格式最好、cap 最少；但算法/边界质量最低，合法代码条件通过率接近 Base。
- **C1：高质量长推理型。**pass 和合法提交后的正确率最高、runtime error 最低、跨 prompt 最稳；代价是 token 与延迟最高，cap 与格式失败仍严重。
- **C2：折中型。**pass 与 C1 相同，成本明显低，cap/格式优于 C1，time-limit 最低；但 Worst 略低，runtime error 高于 C1。

因此训练 prompt 的主要作用是通过 rollout/reward 更新选择一条“能力—终止—成本”行为前沿，而不是简单的“prompt 越详细，分数越高”。同为 step 100 时 C1/C2 比 C0 高 `6.67/6.50pp`，是目前最直接的 train-prompt 证据；C2 的价值主要体现在相对 C1 更好的成本折中，而不是更高 pass。

### 9.2 Test system prompt

Test prompt 的总体影响很弱：

- 对能力：六模型总体跨度仍只有 2pp，18 组模型内配对比较均不显著；
- 对行为：format/cap/tool/tokens/latency 在 trained model 内都近似不变；
- 对匹配：同名 prompt 只有 1pp、不显著的溢价；
- 对个例：能改变具体题目的解法路径和成功状态，但方向不稳定。

因此现有证据支持：**system prompt 的策略信息只有在训练中反复参与 rollout/reward 更新后才产生稳定作用；仅在测试时追加同类说明，没有可靠收益。**

## 10. 建议

### 10.1 当前模型与 test prompt 选择

- **默认 test prompt 建议用 C0。** 它最短，trained models 在 C0 下平均 pass@1 反而最高（38.33%），而更详细 test prompt 没有显著收益。
- **若追求最高跨 prompt 稳健性，选 C1 model。** Average 38.17%、Worst 37.5%，且合法提交后的正确率最高。
- **若兼顾质量与成本，选 C2 model。** 与 C1 pass 持平，但少约 745 completion tokens、快约 9.15 秒，格式/终止也更好。
- **C0 model 适合强调输出可用性/短生成的场景，但应保留 selected step40。** C0-step100 更长、cap 更多且 pass 更低；在当前证据下没有替换 step40 的理由。

### 10.2 下一轮训练最值得做的改动

1. 对 C1/C2 增加 termination-aware shaping：cap penalty、剩余 token budget 特征、在已经得到可用实现后奖励立即提交。
2. 把“合法工具调用”和“是否应该调用工具”拆开；当前几乎 100% 尝试工具，没有学到 C1 设想的 selective tool use。
3. 对 C0 增加“最后一个失败 case”定向数据：高 case-pass wrong-answer、runtime error、time-limit 三类分别采样修复。
4. 重点扩充 hard 题；当前主要收益集中在 easy/medium，hard pass 仍低于 11%。

### 10.3 下一轮实验设计

1. 至少增加 seed 43；当前单 seed 无法判断训练分支排序是否稳定。
2. 下一轮在看 test 前预注册“统一 step 100”和“dev-selected checkpoint”两套对比；这次已补齐 step100，但 post-hoc 不能追溯消除选择偏差。
3. 在 fresh test split 上复验完整 `raw Base → shared SFT → C0/C1/C2 GRPO` 梯子；SFT-only baseline 已补齐，下一步是确认 `+0.83pp` 与 GRPO 增益能否跨 seed/测试集复现。
4. 预注册 primary 与 secondary 指标族；secondary metric 的大量比较目前属于探索性分析，不应用单个未校正 CI 宣称新定律。
5. 对 test prompt 研究可停止扩大复杂度，优先把预算用于多 seed、hard-task 和 termination；六个模型都不支持更长 test prompt。

## 11. 统计与复现说明

- selected final 与 post-hoc 对比均使用每道题配对的 10,000 次 bootstrap，seed `20260813`；二元 pass 翻转使用 exact McNemar。
- trained-vs-Base 九个主比较额外报告 Holm 校正 p-value。
- C0-step100/SFT-only 的比较属于 post-hoc exploratory family；模型内 test-prompt 比较报告 Holm 结果，但没有把复用 test set 包装成新的 confirmatory 显著性结论。
- train/model marginal contrast 先在同一道题的三个 test prompt 上取均值，再 bootstrap 200 个题，避免伪重复。
- secondary metrics 的 CI 是探索性、未做全指标多重比较校正；报告重点依赖效应量、方向一致性和行为机制，而不是追逐单个 p-value。
- 复现脚本：[`03_evaluation/analyze_system_prompt_impact.py`](../../03_evaluation/analyze_system_prompt_impact.py)。
- 机器可读完整统计：[`system-prompt-impact-analysis.json`](../../artifacts/analysis/system-prompt-impact-analysis.json)。
- 原始 final 统计：[`final-evaluation-matrix.md`](../results/final-evaluation-matrix.md)。
- 新增两组的原始汇总：[`posthoc-checkpoint-and-sft-ablation.md`](../results/posthoc-checkpoint-and-sft-ablation.md)。

## 12. 最终判断

0812 实验现在包含完整 200 题、12 个 selected-final cell、6 个 post-hoc cell、全部 0 failed，并可下钻到训练轨迹与 secondary metrics。新增两组使机制归因明显更清楚：

- **共同 SFT 不是 held-out 增益的主要来源。** SFT-only 只比 raw Base 高 `0.83pp`，区间跨 0；后续 GRPO 相对 SFT-only 提高 `5.83–12.50pp`。
- **Train prompt 有真实、可测的权重塑形作用。** 在同为 step 100 时，C1/C2 比 C0 高 `6.67/6.50pp`，且形成不同的工具、长度、终止和错误结构。
- **C0 的 checkpoint 敏感性主要表现为终止退化。** step100 比 selected step40 更长、格式更差、cap 更多，最终低 `2.33pp`；继续训练本身不是保证。
- **Test prompt 没有可靠的总体增益。** 更详细的 C1/C2 在测试时不能替代训练，也没有稳定 matching advantage。
- **最关键的剩余瓶颈不是 prompt 文字本身，而是语义质量与可靠终止之间的冲突。** C0 解决终止但正确性不足；C1/C2 提高正确性但仍过长。下一轮应直接优化这条 trade-off，而不是继续加长 prompt。

在“只看 pass@1”的选择上，C1 与 C2 平手；综合 secondary metrics 后，**C2 + test C0** 是当前最值得优先复验的配置，**C1 + test C0** 是稳健性备选。SFT-only baseline 已经补齐；正式推广前剩下的关键门槛是 **fresh split + 多 seed**，而不是继续补更长的 test prompt。
