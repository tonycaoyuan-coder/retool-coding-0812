# 文档索引

本目录只保存适合交接和审阅的文档。运行产生的轨迹、日志、SQLite、CSV 和 JSON 等机器文件仍位于 `artifacts/`，不会提交到 Git。

## 建议阅读顺序

1. [`code-navigation-guide.md`](code-navigation-guide.md)：代码入口、核心模块和端到端数据流。
2. [`final-experiment-summary.md`](final-experiment-summary.md)：实验设计、完整结果和后续建议。
3. [`experiment-record-and-results.md`](experiment-record-and-results.md)：冻结配置、运行事件和来源记录。

## 结果

- [`results/final-evaluation-matrix.md`](results/final-evaluation-matrix.md)：原始 L10K/20K 的最终 12-cell 矩阵、bootstrap CI 和 McNemar 检验。
- [`results/length-ablation-final-matrix.md`](results/length-ablation-final-matrix.md)：L16K/24K 长度消融的 12-cell 矩阵。
- [`results/posthoc-checkpoint-and-sft-ablation.md`](results/posthoc-checkpoint-and-sft-ablation.md)：C0 step 100 与 shared-SFT-only 的事后对照。

## 专项调查

- [`investigations/token-cap-and-length-ablation-summary.md`](investigations/token-cap-and-length-ablation-summary.md)：触顶失败与长度扩展的综合结论。
- [`investigations/prompt-verbosity-and-cap-hit-analysis.md`](investigations/prompt-verbosity-and-cap-hit-analysis.md)：prompt 详细程度、重复与触顶行为分析。
- [`investigations/system-prompt-impact-analysis.md`](investigations/system-prompt-impact-analysis.md)：训练/测试 system prompt 的统计影响。
- [`investigations/length-ablation-l16k-vs-l10k-report.md`](investigations/length-ablation-l16k-vs-l10k-report.md)：L16K/24K 对 L10K/20K 的完整配对比较。
- [`investigations/length-ablation-analysis-prompt.md`](investigations/length-ablation-analysis-prompt.md)：生成长度消融报告时使用的分析任务说明。
