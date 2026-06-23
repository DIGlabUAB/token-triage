# Paper-Grade TokenTriage Experiment

## Setup

- Dataset: `ChayanM/IUXray-Data-Train-Test` / `test`
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Reports: `64`
- Evaluated tokens: `3783`
- Critical tokens: `341`
- Transformer layers: `24`

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
| entropy | threshold=0.10 | 0.0929 | 0.1144 | 0.1170 | 1.4462 | 0.0876-0.1402 |
| margin | threshold=0.50 | 0.2781 | 0.3754 | 0.3759 | 4.5028 | 0.3259-0.4169 |
| tokentriage-kstable | entropy=0.10,min=0.85,k=4 | 0.0846 | 0.0117 | 0.0793 | 1.3229 | 0.0000-0.0225 |
| tokentriage-kstable | entropy=0.42,min=0.85,k=4 | 0.4788 | 0.1818 | 0.5061 | 6.9197 | 0.1433-0.2281 |
| tokentriage-stable | entropy=0.55,min=0.50 | 0.7914 | 0.0000 | 0.6077 | 9.7295 | 0.0000-0.0000 |

## Top Deployable Pareto Candidates

| Rank | Method | Setting | Compute saved | Critical disagreement | HWD |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | fixed-depth | 1.00 | 0.0000 | 0.0000 | 0.0000 |
| 2 | tokentriage-kstable | entropy=0.10,min=0.85,k=4 | 0.0846 | 0.0117 | 0.0793 |
| 3 | tokentriage-kstable | entropy=0.10,min=0.85,k=3 | 0.0851 | 0.0264 | 0.0848 |
| 4 | tokentriage-kstable | entropy=0.15,min=0.85,k=4 | 0.1372 | 0.0352 | 0.1455 |
| 5 | tokentriage-kstable | entropy=0.10,min=0.72,k=4 | 0.0851 | 0.0381 | 0.0890 |
| 6 | tokentriage-kstable | entropy=0.20,min=0.85,k=4 | 0.1807 | 0.0440 | 0.1890 |
| 7 | tokentriage-kstable | entropy=0.10,min=0.65,k=4 | 0.0859 | 0.0557 | 0.0953 |
| 8 | tokentriage-kstable | entropy=0.10,min=0.72,k=3 | 0.0859 | 0.0616 | 0.0977 |
| 9 | tokentriage-kstable | entropy=0.30,min=0.85,k=4 | 0.2983 | 0.0674 | 0.3065 |
| 10 | tokentriage-kstable | entropy=0.10,min=0.85,k=2 | 0.0856 | 0.0674 | 0.0998 |
| 11 | tokentriage-kstable | entropy=0.10,min=0.50,k=4 | 0.0869 | 0.0704 | 0.1007 |
| 12 | tokentriage-kstable | entropy=0.15,min=0.85,k=3 | 0.1379 | 0.0733 | 0.1593 |

## Interpretation

This experiment compares deployable TokenTriage policies against confidence-style early exits (entropy and margin), fixed-depth execution, and oracle stability upper bounds on real public radiology text. The paper-grade claim is the Pareto shape: policies that optimize confidence alone can retain high compute savings while leaving high disagreement on clinically critical tokens; TokenTriage variants spend extra layers selectively on high-risk tokens and reduce harm-weighted disagreement.

The current metric is final-model preservation, not radiologist correctness. That is appropriate for adaptive-inference evaluation, but a clinical paper should add RadGraph or radiologist-labeled entity correctness before making safety claims about patient outcomes.
