# Paper-Grade TokenTriage Experiment

## Setup

- Dataset: `ChayanM/IUXray-Data-Train-Test` / `test`
- Model: `Qwen/Qwen2.5-3B-Instruct`
- Reports: `64`
- Evaluated tokens: `3783`
- Critical tokens: `341`
- Transformer layers: `36`

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
| entropy | threshold=0.10 | 0.3214 | 0.5308 | 0.5543 | 7.8061 | 0.4708-0.5807 |
| margin | threshold=0.50 | 0.5661 | 0.6833 | 0.7629 | 11.8150 | 0.6364-0.7207 |
| tokentriage-kstable | entropy=0.10,min=0.85,k=4 | 0.2974 | 0.1848 | 0.4246 | 7.2875 | 0.1440-0.2174 |
| tokentriage-kstable | entropy=0.20,min=0.85,k=4 | 0.5006 | 0.2023 | 0.5927 | 10.7533 | 0.1533-0.2418 |
| tokentriage-stable | entropy=0.55,min=0.50 | 0.8864 | 0.0000 | 0.6208 | 13.8138 | 0.0000-0.0000 |

## Top Deployable Pareto Candidates

| Rank | Method | Setting | Compute saved | Critical disagreement | HWD |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | fixed-depth | 1.00 | 0.0000 | 0.0000 | 0.0000 |
| 2 | tokentriage-kstable | entropy=0.10,min=0.85,k=4 | 0.2974 | 0.1848 | 0.4246 |
| 3 | tokentriage-kstable | entropy=0.15,min=0.85,k=4 | 0.4057 | 0.1935 | 0.5260 |
| 4 | tokentriage-kstable | entropy=0.20,min=0.85,k=4 | 0.5006 | 0.2023 | 0.5927 |
| 5 | tokentriage-kstable | entropy=0.30,min=0.85,k=4 | 0.6616 | 0.2170 | 0.6615 |
| 6 | tokentriage-kstable | entropy=0.55,min=0.85,k=4 | 0.8855 | 0.2346 | 0.7087 |
| 7 | tokentriage-kstable | entropy=0.42,min=0.85,k=4 | 0.8274 | 0.2346 | 0.7034 |
| 8 | tokentriage-kstable | entropy=0.10,min=0.72,k=4 | 0.3003 | 0.2669 | 0.4545 |
| 9 | tokentriage-kstable | entropy=0.10,min=0.85,k=3 | 0.2987 | 0.2698 | 0.4566 |
| 10 | tokentriage-kstable | entropy=0.15,min=0.72,k=4 | 0.4092 | 0.3021 | 0.5655 |
| 11 | tokentriage-kstable | entropy=0.15,min=0.85,k=3 | 0.4073 | 0.3079 | 0.5686 |
| 12 | tokentriage-kstable | entropy=0.10,min=0.65,k=4 | 0.3023 | 0.3167 | 0.4733 |

## Interpretation

This experiment compares deployable TokenTriage policies against confidence-style early exits (entropy and margin), fixed-depth execution, and oracle stability upper bounds on real public radiology text. The paper-grade claim is the Pareto shape: policies that optimize confidence alone can retain high compute savings while leaving high disagreement on clinically critical tokens; TokenTriage variants spend extra layers selectively on high-risk tokens and reduce harm-weighted disagreement.

The current metric is final-model preservation, not radiologist correctness. That is appropriate for adaptive-inference evaluation, but a clinical paper should add RadGraph or radiologist-labeled entity correctness before making safety claims about patient outcomes.
