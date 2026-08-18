# L16K/24K Final Evaluation 深度分析任务

你正在分析 retool-coding-0812 的 post-hoc length ablation。请只做数据读取、分析和报告写作，不修改评测代码、配置、SQLite、原始轨迹或既有分析产物。

## 输入

- 新实验目录：`artifacts/evaluation-l16k-t24k/`
- 新实验机器可读分析：`artifacts/analysis/l16k-t24k/`
- 原 L10K/20K 实验目录：`artifacts/evaluation/`
- 原报告：`docs/investigations/prompt-verbosity-and-cap-hit-analysis.md`
- 原机器可读指标：
  - `artifacts/analysis/prompt-verbosity-analysis.json`
  - `artifacts/analysis/prompt-verbosity-cell-metrics.csv`
  - `artifacts/analysis/system-prompt-impact-analysis.json`

## 必须先做的验证

1. 读取新实验的 `audit.json`，确认 2,400/2,400 completed、12 个 cell 完整、三组 prompt 使用同一批 200 个 sample ID，且逐条 token/logprob/count 对齐。
2. 核对新旧配置：模型/checkpoint、seed 42、P1、greedy decoding、冻结测试集、prompt 文本和除长度外的协议参数一致；明确记录 L10K/20K 与 L16K/24K 的唯一预期差异。
3. 若审计未通过或数据不完整，不得编造结论；将问题写入目标报告的“数据完整性与限制”，并让最终消息明确说明阻塞。

## 分析要求

围绕 Base、C0@40、C1@100、C2@100 × C0/C1/C2 test prompt 的 12 个 cell，对比 L10K/20K 与 L16K/24K：

- pass@1、case/public/private pass rate，以及 format-valid、compile/runtime/time-limit rate。
- 总体/第一轮 cap-hit，cap 与格式失败、无 final、重复生成的关联。
- tool attempt/use/valid rate、mean tool calls、mean turns；解释报告中的第一轮、第二轮工具调用口径。
- prompt、completion、trajectory tokens；completion 均值/中位数、第一轮 tokens、unsubmitted tokens、final-turn share。
- 重复行率、重复 8-gram、改口标记/1K、cap-only 重复行率。
- judge execution time、完整轨迹 latency。
- pass given valid、pass given tool/no-tool。
- 按模型、test prompt、难度、平台分层。
- 逐题配对：pass/format/cap 双向 flip、exact McNemar；连续指标 paired bootstrap 95% CI。
- 原先 cap-hit 的题中转化为合法提交和正确提交的比例；增长预算后仍触顶、仍重复、仍无 final 的比例。
- 将变化拆解为“延长预算挽救成功”和“只增加无效生成成本”，避免把相关性写成因果。
- 识别整体平均掩盖的异质性，以及 Base 与三个 checkpoint 在不同 prompt 下的差异。
- 选择少量有代表性的原始轨迹作为案例，给出 instance/model/prompt 标识和可复核证据；不得复制隐藏测试内容。

所有关键数字应能追溯到 JSON/CSV/SQLite/trajectory 文件。优先复用已有机器可读结果，并针对关键结论做独立交叉核验。区分百分比与百分点，说明分母；不对非显著差异做强结论。

## 输出

生成一份结构清晰、详细但不重复堆砌的中文 Markdown 报告：

`docs/investigations/length-ablation-l16k-vs-l10k-report.md`

报告至少包含：执行摘要、实验与可比性、数据完整性、总体结果、12-cell 结果、逐题配对推断、长度/触顶机制、工具使用、重复与无效生成、效率与时延、分层与代表案例、威胁与限制、结论与建议。附录列出关键输入文件和指标定义。

最终回复只需说明报告路径、审计是否通过、最重要的 5 条结论；不要在最终回复中粘贴整篇报告。
