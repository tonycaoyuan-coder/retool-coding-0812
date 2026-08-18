# ReTool-Coding 0812：Prompt 冗长、重复与 Cap-hit 对比报告

> 分析对象：最终评测与 post-hoc 最终评测的 3,600 条原始轨迹，seed 42。
> 每个 Train × Test cell 含同一批 200 道 LCB-v6 题；生成使用 greedy decoding。
> 数据来源：`artifacts/evaluation` 与 `artifacts/evaluation-posthoc` 下逐轨迹 `.json.gz`；统计生成时间：2026-08-14。

## 0. 结论先行

这组数据把“prompt 详细导致废话和 cap-hit”拆成两个不同问题：

1. **测试时临时换成更详细的 C2 prompt，并没有让入选模型总体变长。** 固定 C0/C1/C2 三个入选权重后，test C0 与 test C2 的平均 completion 分别为 `4687.4` 与 `4692.1` tokens，差值只有 `+4.8`；cap-hit 反而从 `28.33%` 变为 `26.83%`。
2. **训练时使用更详细的 prompt，与更长、更易触顶的权重行为明显相关。** 跨三个 test prompt，C0@100 / C1@100 / C2@100 的平均 completion 为 `4542` / `5624` / `4880` tokens，cap-hit 为 `27.00%` / `35.83%` / `30.50%`。其中 C1 的关联最强。
3. **确实存在由重复或绕路直接导致触顶的原始样例，但不是所有额外 token 都是废话。** 例如 `Humidifier 1` 的同模型配对中，C0 用 865 tokens 通过，C2 在第一轮生成到 10,240 tokens，出现高度重复后被截断；但 `Adjacent GCD` 存在相反方向，C0 触顶而 C2 用 5,561 tokens 通过。
4. **cap-hit 主要发生在第一次 assistant turn。** `1051/1080`（`97.31%`）的 cap-hit 在第一轮发生，通常尚未完成合法工具调用和最终提交。这更像是模型把推理、草稿或循环内容塞进第一次 tool-call payload，而不是看完工具结果后才耗尽预算。
5. **当前证据支持的是行为关联，不是跨 seed 的训练因果定律。** 统一 step100 能排除 checkpoint 步数混杂，但仍只有一个训练 seed；‘废话’指标也只是自动代理，需要结合盲审和强制收束反事实实验。

## 1. 数据与指标口径

### 1.1 数据完整性

- selected final：Base、C0@40、C1@100、C2@100 × C0/C1/C2 test prompt，共 2,400 条。
- post-hoc final：SFT-only、C0@100 × C0/C1/C2 test prompt，共 1,200 条。
- 合计 18 个 cell、3,600 条轨迹；每个 cell 200 条，键为 `model × system_prompt_id × instance_id`，无重复或缺失。
- 只分析 `turns[].text` 与 `turns[].completion_token_count`，不把 system/user prompt 或 tool observation 计入模型冗长与重复。

### 1.2 三组可测指标

| 指标组 | 本报告的量化定义 | 解释边界 |
|---|---|---|
| 冗长度 | completion token 均值/中位数、第一轮 tokens、未进入合法最终提交的 tokens、最终提交轮 token 占比 | `未提交 tokens`：有合法 final 时为此前各轮 tokens；无合法 final 时为全部生成 tokens。它衡量未交付生成，不等于全部无用。 |
| 重复与绕路 | 每条轨迹的轮内重复行率、重复 8-gram 率、每千 token 的改口标记次数；另报告 cap 轨迹的重复行率 | 重复只在同一 assistant turn 内计算，避免把工具测试代码复制为 final code 误判为重复。改口标记匹配 `wait/actually/let's try/different approach/not right/rethink` 等短语，是启发式代理。 |
| 最终代价 | cap-hit、第一轮 cap-hit、format-valid、pass@1、合法提交条件通过率、轨迹时延 | pass 和格式来自本地 judge；时延可能受服务负载影响，不作跨批严格因果排序。 |

重复指标不能单独证明‘废话’：正确程序中的循环结构、公式和必要复核也会重复。最强证据应是‘高重复/多次改口 + 未形成 final + 强制提前收束后正确率恢复’的组合；最后一项需要新增反事实评测。

## 2. 18-cell 冗长度指标

| Train model | Test | Mean completion | Median completion | Mean first turn | Mean unsubmitted | Final-turn share |
|---|---|---:|---:|---:|---:|---:|
| Base | C0 | 4,414.0 | 1,653.0 | 4,223.0 | 4,221.0 | 4.4% |
| Base | C1 | 4,568.3 | 1,560.0 | 4,199.0 | 4,182.5 | 8.4% |
| Base | C2 | 4,707.5 | 1,861.0 | 4,436.1 | 4,433.1 | 5.8% |
| SFT-only | C0 | 4,189.3 | 1,517.5 | 3,897.8 | 3,895.3 | 7.0% |
| SFT-only | C1 | 4,150.5 | 1,374.5 | 3,644.0 | 3,635.9 | 12.4% |
| SFT-only | C2 | 4,243.7 | 1,659.0 | 3,901.5 | 3,899.4 | 8.1% |
| C0@40 | C0 | 3,526.3 | 1,871.0 | 3,053.3 | 3,053.3 | 13.4% |
| C0@40 | C1 | 3,854.4 | 1,925.0 | 3,468.9 | 3,468.9 | 10.0% |
| C0@40 | C2 | 3,547.8 | 2,093.5 | 3,214.6 | 3,214.6 | 9.4% |
| C0@100 | C0 | 4,386.1 | 2,459.5 | 4,088.0 | 4,088.0 | 6.8% |
| C0@100 | C1 | 4,582.0 | 2,610.0 | 4,242.4 | 4,242.4 | 7.4% |
| C0@100 | C2 | 4,657.8 | 2,971.5 | 4,304.7 | 4,304.7 | 7.6% |
| C1@100 | C0 | 5,709.4 | 4,300.5 | 5,375.3 | 5,375.3 | 5.9% |
| C1@100 | C1 | 5,497.6 | 4,432.0 | 5,106.0 | 5,106.0 | 7.1% |
| C1@100 | C2 | 5,666.5 | 5,060.0 | 5,350.7 | 5,350.7 | 5.6% |
| C2@100 | C0 | 4,826.4 | 2,983.5 | 4,558.2 | 4,558.2 | 5.6% |
| C2@100 | C1 | 4,950.3 | 3,760.0 | 4,632.2 | 4,632.2 | 6.4% |
| C2@100 | C2 | 4,862.1 | 3,120.5 | 4,601.7 | 4,601.7 | 5.4% |

读表重点：

- C0@40 在三个 test prompt 下都最短；C1@100 都最长。
- `Mean unsubmitted` 同时受到‘第一轮工具调用较长’与‘触顶后完全没有 final’影响，因此它比总 token 更接近终止失败成本。
- 同一模型横向看 C0/C1/C2 test prompt，没有出现一致的‘越详细越长’阶梯；纵向看训练权重，则 C1/C2，尤其 C1，明显比 C0 更长。

## 3. 18-cell 重复与绕路代理指标

| Train model | Test | Repeated lines | Repeated 8-grams | Revision markers / 1K | Cap-only repeated lines |
|---|---|---:|---:|---:|---:|
| Base | C0 | 26.3% | 29.9% | 3.13 | 67.4% |
| Base | C1 | 25.7% | 29.2% | 3.16 | 60.7% |
| Base | C2 | 31.4% | 34.5% | 3.65 | 72.9% |
| SFT-only | C0 | 25.1% | 28.8% | 3.20 | 70.2% |
| SFT-only | C1 | 24.0% | 27.5% | 2.89 | 63.9% |
| SFT-only | C2 | 24.3% | 27.6% | 3.41 | 66.9% |
| C0@40 | C0 | 12.0% | 16.6% | 1.77 | 49.4% |
| C0@40 | C1 | 11.7% | 16.5% | 1.71 | 40.5% |
| C0@40 | C2 | 12.3% | 16.9% | 1.72 | 48.2% |
| C0@100 | C0 | 17.7% | 22.3% | 1.87 | 52.8% |
| C0@100 | C1 | 19.5% | 24.3% | 1.91 | 50.3% |
| C0@100 | C2 | 19.9% | 24.8% | 2.07 | 56.0% |
| C1@100 | C0 | 26.3% | 31.2% | 3.62 | 59.2% |
| C1@100 | C1 | 23.6% | 29.1% | 3.53 | 56.9% |
| C1@100 | C2 | 24.7% | 30.8% | 3.48 | 58.8% |
| C2@100 | C0 | 9.6% | 15.8% | 2.21 | 24.2% |
| C2@100 | C1 | 9.0% | 16.4% | 2.17 | 20.2% |
| C2@100 | C2 | 9.8% | 17.3% | 2.32 | 24.3% |

这些平均值会被少量极端循环显著拉高，因此报告同时保留原始轨迹案例。重复率较低也不代表没有废话：模型可能不断提出不同但均失败的算法，形成长而不重复的 dead-end chain，例如下文的 `Double Sum 2` C2 轨迹。

## 4. 18-cell 最终代价

| Train model | Test | Cap-hit | First-turn cap | Format-valid | Pass@1 | Pass given valid | Mean latency |
|---|---|---:|---:|---:|---:|---:|---:|
| Base | C0 | 35.0% | 35.0% | 65.0% | 25.0% | 38.5% | 39.5s |
| Base | C1 | 38.5% | 36.5% | 63.5% | 24.0% | 37.8% | 40.9s |
| Base | C2 | 39.5% | 38.5% | 61.5% | 25.5% | 41.5% | 43.4s |
| SFT-only | C0 | 32.0% | 31.0% | 69.0% | 25.5% | 37.0% | 79.7s |
| SFT-only | C1 | 32.5% | 29.5% | 70.5% | 25.0% | 35.5% | 77.2s |
| SFT-only | C2 | 32.5% | 31.0% | 69.0% | 26.5% | 38.4% | 78.8s |
| C0@40 | C0 | 15.5% | 14.0% | 86.0% | 36.0% | 41.9% | 55.3s |
| C0@40 | C1 | 18.5% | 18.0% | 82.0% | 34.0% | 41.5% | 60.9s |
| C0@40 | C2 | 16.0% | 16.0% | 84.0% | 31.5% | 37.5% | 56.1s |
| C0@100 | C0 | 26.5% | 26.5% | 73.5% | 33.5% | 45.6% | 82.9s |
| C0@100 | C1 | 27.5% | 27.0% | 73.0% | 32.5% | 44.5% | 87.8s |
| C0@100 | C2 | 27.0% | 26.5% | 73.5% | 28.5% | 38.8% | 88.4s |
| C1@100 | C0 | 39.5% | 38.5% | 61.5% | 38.5% | 62.6% | 88.2s |
| C1@100 | C1 | 34.0% | 33.0% | 67.0% | 38.5% | 57.5% | 84.6s |
| C1@100 | C2 | 34.0% | 33.5% | 66.5% | 37.5% | 56.4% | 87.8s |
| C2@100 | C0 | 30.0% | 30.0% | 70.0% | 40.5% | 57.9% | 77.2s |
| C2@100 | C1 | 31.0% | 30.5% | 69.5% | 36.0% | 51.8% | 78.7s |
| C2@100 | C2 | 30.5% | 30.5% | 69.5% | 37.5% | 54.0% | 77.4s |

最终代价呈现清晰的 trade-off：C0 权重更容易及时交付，但合法代码的条件正确率较低；C1/C2 权重生成更长、触顶更多，但一旦合法提交，正确率明显更高。因此优化目标不能只是缩短输出，而应减少‘未形成 final 的冗长’，保留能提高代码正确性的有效推理。

## 5. 固定模型与题目：Test C2 − Test C0 配对

每行使用同一模型的同一 200 道题做配对。token 和重复率报告 C2−C0 的均值与逐题 bootstrap 95% CI；cap flips 中 `C2-only` 表示只有 C2 触顶，`C0-only` 表示只有 C0 触顶。

| Model | Δ completion tokens | Δ unsubmitted tokens | Δ repeated lines | Δ cap-hit | Cap flips C0-only / C2-only |
|---|---:|---:|---:|---:|---:|
| Base | +293 [-183, +774] | +212 [-291, +706] | +5.1pp [+0.4, +9.9] | +4.5pp [-1.5, +11.0] | 15 / 24 |
| SFT-only | +54 [-376, +504] | +4 [-449, +464] | -0.8pp [-5.0, +3.2] | +0.5pp [-4.5, +6.0] | 14 / 15 |
| C0@40 | +22 [-376, +412] | +161 [-247, +580] | +0.3pp [-2.5, +3.1] | +0.5pp [-4.0, +5.0] | 11 / 12 |
| C0@100 | +272 [-149, +679] | +217 [-223, +664] | +2.2pp [-1.3, +5.8] | +0.5pp [-6.0, +7.0] | 20 / 21 |
| C1@100 | -43 [-447, +369] | -25 [-434, +390] | -1.6pp [-5.6, +2.2] | -5.5pp [-12.0, +1.0] | 29 / 18 |
| C2@100 | +36 [-263, +331] | +43 [-274, +352] | +0.3pp [-1.9, +2.6] | +0.5pp [-4.0, +5.0] | 10 / 11 |

在三个入选模型上聚合，test C2 相对 test C0 只多 `+4.8` tokens，cap-hit 低 `1.50pp`。在全部六个权重上，C2 平均多 `+105.6` tokens，cap-hit 只高 `+0.17pp`。这些结果不支持 test-time 详细 prompt 是总体 cap 增长的主因。

同时，各模型都有 C0-only 与 C2-only cap 翻转，说明 prompt 会改变个别题的生成路径；应报告双向案例，而不能只展示支持猜想的样本。

## 6. 统一 Step 100：Train prompt 对比

下表固定同一 test prompt，并用同一 200 题比较 C1@100/C2@100 与 C0@100。这样排除了 C0 selected checkpoint 为 step40 的步数混杂。CI 仍只反映题目抽样不确定性，不包含训练 seed 方差。

| Train contrast | Test | Δ completion tokens | Δ unsubmitted | Δ repeated lines | Δ cap-hit | Δ pass@1 |
|---|---|---:|---:|---:|---:|---:|
| C1@100−C0@100 | C0 | +1323 [+896, +1749] | +1287 [+828, +1753] | +8.7pp [+4.8, +12.6] | +13.0pp [+6.5, +19.5] | +5.0pp [+0.5, +10.0] |
| C2@100−C0@100 | C0 | +440 [+32, +862] | +470 [+23, +921] | -8.1pp [-11.6, -4.7] | +3.5pp [-2.5, +9.5] | +7.0pp [+1.0, +13.0] |
| C1@100−C0@100 | C1 | +916 [+473, +1381] | +864 [+367, +1371] | +4.1pp [+0.3, +8.0] | +6.5pp [+0.0, +13.5] | +6.0pp [+1.5, +10.5] |
| C2@100−C0@100 | C1 | +368 [-65, +804] | +390 [-51, +836] | -10.5pp [-14.0, -7.0] | +3.5pp [-2.5, +9.5] | +3.5pp [-1.5, +8.5] |
| C1@100−C0@100 | C2 | +1009 [+548, +1470] | +1046 [+567, +1530] | +4.8pp [+1.1, +8.7] | +7.0pp [+0.0, +13.5] | +9.0pp [+3.5, +14.5] |
| C2@100−C0@100 | C2 | +204 [-261, +671] | +297 [-188, +802] | -10.0pp [-13.9, -6.2] | +3.5pp [-3.0, +10.0] | +9.0pp [+3.5, +14.5] |

统一 step 后，C1 相对 C0 的长度与 cap 增量仍最大；C2 的增量更小。这支持‘训练 prompt 塑造了终止/长度行为’，并与 C2 相对 C1 更好的成本折中一致。不过 C1/C2 同时提高 pass，因此额外 token 中既有有效推理，也有终止失败，不能一概删除。

## 7. 原始轨迹对比

### 7.1 支持猜想的同模型样例：Humidifier 1

同一个 C0@100 权重、同一道题：[`test C0` 原始轨迹](../../artifacts/evaluation-posthoc/c0/20260813T132142Z-retool-coding-0812-posthoc-c0-seed42-9cd108c3/artifacts/trajectories/a88a224db6b787632fdf7d82610d02779930872c074dce144b930db294df6b0a.json.gz) 使用 `865` tokens，合法提交并通过；[`test C2` 原始轨迹](../../artifacts/evaluation-posthoc/c2/20260813T132142Z-retool-coding-0812-posthoc-c2-seed42-50526731/artifacts/trajectories/e0b997be12fa9820d641ee619d4b8a067a36706bd477be3ec2ae57ebcbe04aef.json.gz) 在第一轮达到 `10240` tokens，未形成合法 final。

C2 轨迹的轮内重复行率为 `93.1%`。最高频内容包括：

- `by keeping track of the last time`：220 次
- `let's track the time difference from the last operation`：113 次
- `actually, we can just track the time difference from the last operation`：111 次
- `initialize last_time = 0`：110 次
- `let's use a variable to track the last time`：109 次
- `but we can't do that in the loop without storing it`：109 次
- `track the time difference from the last operation`：2 次
- `and subtract it from water (if water > 0)`：2 次

这条轨迹可以明确归入‘重复循环导致 cap-hit’，而不只是算法本身复杂。

### 7.2 反方向样例：Adjacent GCD

同一个 C0@100 权重、同一道题：[`test C0` 原始轨迹](../../artifacts/evaluation-posthoc/c0/20260813T132142Z-retool-coding-0812-posthoc-c0-seed42-9cd108c3/artifacts/trajectories/027d52f60e3c19cf00b864582cb9d4e71df283dc513874e00ea12330b9f4929f.json.gz) 在第一轮 10,240 tokens 触顶；[`test C2` 原始轨迹](../../artifacts/evaluation-posthoc/c2/20260813T132142Z-retool-coding-0812-posthoc-c2-seed42-50526731/artifacts/trajectories/c8134d468bfdcc03682a053abf90206022e894aac9b6c45f432aa12154709b11.json.gz) 使用 5,561 tokens，形成合法提交并通过。它说明详细 prompt 有时也会把模型从发散路径切换到可完成路径。

### 7.3 固定 Test C0 的训练分支样例：Double Sum 2

在同一道题、同一个 test C0 下，[`C0@40`](../../artifacts/evaluation/c0/20260813T020037Z-retool-coding-0812-evaluation-c0-seed42-8f3ac38e/artifacts/trajectories/ed13b8e19ff1e291af09a62b87789efca35b974c126d5088d1fc8c0132709293.json.gz) 用 `492` tokens 提交并通过；[`C1@100`](../../artifacts/evaluation/c0/20260813T020037Z-retool-coding-0812-evaluation-c0-seed42-8f3ac38e/artifacts/trajectories/90a90f4e212addf837046bd011a6448e7b0a6f5fadcf0963ec025f78bf175952.json.gz) 和 [`C2@100`](../../artifacts/evaluation/c0/20260813T020037Z-retool-coding-0812-evaluation-c0-seed42-8f3ac38e/artifacts/trajectories/96e5a76be1574f6565de0e5e104f46b1ca92a55413cb58239ed5c01acd8debf6.json.gz) 均在第一轮达到 10,240 tokens 后失败。C1 的重复行率为 `60.5%`；C2 的重复率较低（`19.3%`），但连续尝试多个未完成方向。这展示了两种不同的‘废话’：显式循环，以及不重复但持续换路的 dead-end chain。

该案例仍不是训练 prompt 的独立统计证据；总体判断应以前一节的 200 题配对和多 seed 复验为准。

## 8. 如何把猜想升级为更强的验证

1. **盲审分层样本。** 从 C2-only cap、C0-only cap、both-cap、neither-cap 各抽取相同数量轨迹，隐藏模型和 prompt，标注 problem restatement、重复推导、算法换路、工具草稿过长、已有可用代码但未提交等类别；报告双标注一致率。
2. **强制收束反事实。** 固定现有 checkpoint 和题目，在 4K/6K/8K tokens 或检测到高重复时注入‘立即提交当前最佳代码’，与原轨迹做同题配对。若 cap 降低且 pass 不降，才说明被删掉的 token 主要是无用冗长。
3. **训练消融。** 固定 C2 策略，比较当前 reward、termination cue、termination-aware shaping 及二者组合；先做 20–40 step canary。主指标应同时包含 pass@1、cap-hit、未提交 tokens 和合法提交条件通过率。
4. **多 seed。** 当前训练 prompt 对比只有 seed 42。至少补 seed 43/44，并在同一 seed 内让各分支共享初始化、训练题顺序和采样随机流。

## 9. 复现文件

- 分析脚本：[`03_evaluation/analyze_prompt_verbosity.py`](../../03_evaluation/analyze_prompt_verbosity.py)
- 18-cell CSV：[`prompt-verbosity-cell-metrics.csv`](../../artifacts/analysis/prompt-verbosity-cell-metrics.csv)
- 完整机器可读结果：[`prompt-verbosity-analysis.json`](../../artifacts/analysis/prompt-verbosity-analysis.json)
- 原始轨迹：`artifacts/evaluation/*/*/artifacts/trajectories/*.json.gz` 与 `artifacts/evaluation-posthoc/*/*/artifacts/trajectories/*.json.gz`。

统计使用每道题配对的 10,000 次 bootstrap，seed `20260814`。自动重复指标按 assistant turn 分段，tool observation 不进入统计。
