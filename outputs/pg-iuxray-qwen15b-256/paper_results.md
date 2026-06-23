# Paper-Grade TokenTriage Experiment

## Setup

- Dataset: `ChayanM/IUXray-Data-Train-Test` / `test`
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Reports: `256`
- Evaluated tokens: `13697`
- Critical tokens: `1163`
- Transformer layers: `28`

## Critical Token Mix

- finding: 125
- laterality: 238
- measurement: 19
- measurement_unit: 5
- negation: 419
- severity: 82
- temporality: 275

## Strongest Baseline Comparison

| Method | Setting | Compute saved | Critical disagreement | HWD | Excess NLL | 95% CI critical |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| entropy | threshold=0.10 | 0.3789 | 0.7171 | 0.5598 | 5.8341 | 0.6862-0.7471 |
| margin | threshold=0.50 | 0.5714 | 0.8650 | 0.7790 | 8.2471 | 0.8480-0.8840 |
| tokentriage-kstable | entropy=0.10,min=0.85,k=4 | 0.3310 | 0.2485 | 0.3898 | 5.2718 | 0.2258-0.2762 |
| tokentriage-kstable | entropy=0.20,min=0.85,k=4 | 0.5165 | 0.2700 | 0.5753 | 7.5375 | 0.2472-0.2971 |
| tokentriage-stable | entropy=0.55,min=0.50 | 0.8735 | 0.0000 | 0.6304 | 10.5790 | 0.0000-0.0000 |

## Top Deployable Pareto Candidates

| Rank | Method | Setting | Compute saved | Critical disagreement | HWD |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | fixed-depth | 1.00 | 0.0000 | 0.0000 | 0.0000 |
| 2 | tokentriage-kstable | entropy=0.10,min=0.85,k=4 | 0.3310 | 0.2485 | 0.3898 |
| 3 | tokentriage-kstable | entropy=0.15,min=0.85,k=4 | 0.4384 | 0.2605 | 0.5028 |
| 4 | tokentriage-kstable | entropy=0.20,min=0.85,k=4 | 0.5165 | 0.2700 | 0.5753 |
| 5 | tokentriage-kstable | entropy=0.30,min=0.85,k=4 | 0.6448 | 0.2777 | 0.6669 |
| 6 | tokentriage-kstable | entropy=0.42,min=0.85,k=4 | 0.7774 | 0.2794 | 0.7113 |
| 7 | tokentriage-kstable | entropy=0.55,min=0.85,k=4 | 0.8731 | 0.2846 | 0.7297 |
| 8 | tokentriage-kstable | entropy=0.10,min=0.85,k=3 | 0.3322 | 0.3414 | 0.4231 |
| 9 | tokentriage-kstable | entropy=0.10,min=0.72,k=4 | 0.3351 | 0.3525 | 0.4267 |
| 10 | tokentriage-kstable | entropy=0.15,min=0.85,k=3 | 0.4398 | 0.3680 | 0.5413 |
| 11 | tokentriage-kstable | entropy=0.15,min=0.72,k=4 | 0.4429 | 0.3792 | 0.5450 |
| 12 | tokentriage-kstable | entropy=0.20,min=0.85,k=3 | 0.5181 | 0.3921 | 0.6190 |

## Interpretation

This experiment compares deployable TokenTriage policies against confidence-style early exits (entropy and margin), fixed-depth execution, and oracle stability upper bounds on real public radiology text. The paper-grade claim is the Pareto shape: policies that optimize confidence alone can retain high compute savings while leaving high disagreement on clinically critical tokens; TokenTriage variants spend extra layers selectively on high-risk tokens and reduce harm-weighted disagreement.

The current metric is final-model preservation, not radiologist correctness. That is appropriate for adaptive-inference evaluation, but a clinical paper should add RadGraph or radiologist-labeled entity correctness before making safety claims about patient outcomes.
