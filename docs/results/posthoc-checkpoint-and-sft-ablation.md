# ReTool-Coding-0812 Post-hoc Evaluation

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
