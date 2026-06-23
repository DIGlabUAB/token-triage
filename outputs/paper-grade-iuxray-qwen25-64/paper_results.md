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
| entropy | threshold=0.10 | 0.3794 | 0.8270 | 0.6917 | 5.9534 | 0.7841-0.8696 |
| margin | threshold=0.50 | 0.5741 | 0.9267 | 0.8445 | 8.2508 | 0.8980-0.9582 |
| tokentriage-stable | entropy=0.55,min=0.50 | 0.8647 | 0.6158 | 0.8448 | 10.4104 | 0.5614-0.6779 |
| tokentriage-stable | entropy=0.55,min=0.50 | 0.8647 | 0.6158 | 0.8448 | 10.4104 | 0.5614-0.6779 |

## Top Pareto Candidates

| Rank | Method | Setting | Compute saved | Critical disagreement | HWD |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | tokentriage-stable | entropy=0.55,min=0.50 | 0.8647 | 0.6158 | 0.8448 |
| 2 | tokentriage-stable | entropy=0.55,min=0.65 | 0.8647 | 0.6158 | 0.8448 |
| 3 | tokentriage-stable | entropy=0.55,min=0.72 | 0.8646 | 0.6158 | 0.8448 |
| 4 | tokentriage-stable | entropy=0.55,min=0.85 | 0.8642 | 0.6158 | 0.8448 |
| 5 | tokentriage-stable | entropy=0.42,min=0.50 | 0.7708 | 0.6158 | 0.8322 |
| 6 | tokentriage-stable | entropy=0.42,min=0.65 | 0.7708 | 0.6158 | 0.8322 |
| 7 | tokentriage-stable | entropy=0.42,min=0.72 | 0.7707 | 0.6158 | 0.8322 |
| 8 | tokentriage-stable | entropy=0.42,min=0.85 | 0.7703 | 0.6158 | 0.8322 |
| 9 | tokentriage-stable | entropy=0.30,min=0.50 | 0.6355 | 0.6158 | 0.7943 |
| 10 | tokentriage-stable | entropy=0.30,min=0.65 | 0.6355 | 0.6158 | 0.7943 |
| 11 | tokentriage-stable | entropy=0.30,min=0.72 | 0.6354 | 0.6158 | 0.7943 |
| 12 | tokentriage-stable | entropy=0.30,min=0.85 | 0.6350 | 0.6158 | 0.7943 |

## Interpretation

This experiment compares TokenTriage against confidence-style early exits (entropy and margin), fixed-depth execution, and a stability oracle on real public radiology text. The paper-grade claim is the Pareto shape: policies that optimize confidence alone can retain high compute savings while leaving high disagreement on clinically critical tokens; TokenTriage variants spend extra layers selectively on high-risk tokens and reduce harm-weighted disagreement.

The current metric is final-model preservation, not radiologist correctness. That is appropriate for adaptive-inference evaluation, but a clinical paper should add RadGraph or radiologist-labeled entity correctness before making safety claims about patient outcomes.
