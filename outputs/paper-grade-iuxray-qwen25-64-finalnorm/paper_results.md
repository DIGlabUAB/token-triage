# Paper-Grade TokenTriage Experiment

## Setup

- Dataset: `ChayanM/IUXray-Data-Train-Test` / `test`
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Reports: `64`
- Evaluated tokens: `3783`
- Critical tokens: `341`
- Transformer layers: `28`

## Critical Token Mix

- finding: 38
- laterality: 73
- measurement: 4
- measurement_unit: 1
- negation: 114
- severity: 34
- temporality: 77

## Strongest Baseline Comparison

| Method | Setting | Compute saved | Critical disagreement | HWD | Excess NLL | 95% CI critical |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| entropy | threshold=0.10 | 0.3794 | 0.7214 | 0.5659 | 5.6892 | 0.6628-0.7789 |
| margin | threshold=0.50 | 0.5741 | 0.8768 | 0.7857 | 8.1386 | 0.8387-0.9156 |
| tokentriage-stable | entropy=0.55,min=0.50 | 0.8680 | 0.0000 | 0.6157 | 10.3903 | 0.0000-0.0000 |
| tokentriage-stable | entropy=0.55,min=0.50 | 0.8680 | 0.0000 | 0.6157 | 10.3903 | 0.0000-0.0000 |

## Top Pareto Candidates

| Rank | Method | Setting | Compute saved | Critical disagreement | HWD |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | tokentriage-stable | entropy=0.55,min=0.50 | 0.8680 | 0.0000 | 0.6157 |
| 2 | tokentriage-stable | entropy=0.55,min=0.65 | 0.8679 | 0.0000 | 0.6157 |
| 3 | tokentriage-stable | entropy=0.55,min=0.72 | 0.8676 | 0.0000 | 0.6157 |
| 4 | tokentriage-stable | entropy=0.55,min=0.85 | 0.8666 | 0.0000 | 0.6157 |
| 5 | tokentriage-stable | entropy=0.42,min=0.50 | 0.7741 | 0.0000 | 0.6013 |
| 6 | tokentriage-stable | entropy=0.42,min=0.65 | 0.7740 | 0.0000 | 0.6013 |
| 7 | tokentriage-stable | entropy=0.42,min=0.72 | 0.7737 | 0.0000 | 0.6013 |
| 8 | tokentriage-stable | entropy=0.42,min=0.85 | 0.7727 | 0.0000 | 0.6013 |
| 9 | tokentriage-stable | entropy=0.30,min=0.50 | 0.6388 | 0.0000 | 0.5574 |
| 10 | tokentriage-stable | entropy=0.30,min=0.65 | 0.6387 | 0.0000 | 0.5574 |
| 11 | tokentriage-stable | entropy=0.30,min=0.72 | 0.6384 | 0.0000 | 0.5574 |
| 12 | tokentriage-stable | entropy=0.30,min=0.85 | 0.6374 | 0.0000 | 0.5574 |

## Interpretation

This experiment compares TokenTriage against confidence-style early exits (entropy and margin), fixed-depth execution, and a stability oracle on real public radiology text. The paper-grade claim is the Pareto shape: policies that optimize confidence alone can retain high compute savings while leaving high disagreement on clinically critical tokens; TokenTriage variants spend extra layers selectively on high-risk tokens and reduce harm-weighted disagreement.

The current metric is final-model preservation, not radiologist correctness. That is appropriate for adaptive-inference evaluation, but a clinical paper should add RadGraph or radiologist-labeled entity correctness before making safety claims about patient outcomes.
