# Paper-Grade TokenTriage Experiment

## Setup

- Dataset: `ChayanM/IUXray-Data-Train-Test` / `test`
- Model: `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- Reports: `64`
- Evaluated tokens: `3721`
- Critical tokens: `349`
- Transformer layers: `24`

## Critical Token Mix

- finding: 44
- laterality: 74
- measurement: 4
- measurement_unit: 1
- negation: 114
- severity: 35
- temporality: 77

## Strongest Baseline Comparison

| Method | Setting | Compute saved | Critical disagreement | HWD | Excess NLL | 95% CI critical |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| entropy | threshold=0.10 | 0.9102 | 0.9914 | 0.9917 | 24.2263 | 0.9814-1.0000 |
| margin | threshold=0.50 | 0.9039 | 0.8911 | 0.9572 | 23.7590 | 0.8558-0.9188 |
| tokentriage-kstable | entropy=0.10,min=0.85,k=4 | 0.8305 | 0.3066 | 0.7259 | 22.3281 | 0.2661-0.3498 |
| tokentriage-kstable | entropy=0.10,min=0.85,k=4 | 0.8305 | 0.3066 | 0.7259 | 22.3281 | 0.2661-0.3498 |
| tokentriage-stable | entropy=0.42,min=0.50 | 0.8716 | 0.0000 | 0.6168 | 20.8868 | 0.0000-0.0000 |

## Top Deployable Pareto Candidates

| Rank | Method | Setting | Compute saved | Critical disagreement | HWD |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | fixed-depth | 1.00 | 0.0000 | 0.0000 | 0.0000 |
| 2 | tokentriage-kstable | entropy=0.10,min=0.85,k=4 | 0.8305 | 0.3066 | 0.7259 |
| 3 | tokentriage-kstable | entropy=0.42,min=0.85,k=4 | 0.8723 | 0.3095 | 0.7321 |
| 4 | tokentriage-kstable | entropy=0.55,min=0.85,k=4 | 0.8723 | 0.3095 | 0.7321 |
| 5 | tokentriage-kstable | entropy=0.30,min=0.85,k=4 | 0.8720 | 0.3095 | 0.7321 |
| 6 | tokentriage-kstable | entropy=0.20,min=0.85,k=4 | 0.8692 | 0.3095 | 0.7321 |
| 7 | tokentriage-kstable | entropy=0.15,min=0.85,k=4 | 0.8608 | 0.3095 | 0.7319 |
| 8 | tokentriage-kstable | entropy=0.10,min=0.85,k=3 | 0.8325 | 0.4728 | 0.7897 |
| 9 | tokentriage-kstable | entropy=0.42,min=0.85,k=3 | 0.8743 | 0.4756 | 0.7959 |
| 10 | tokentriage-kstable | entropy=0.55,min=0.85,k=3 | 0.8743 | 0.4756 | 0.7959 |
| 11 | tokentriage-kstable | entropy=0.30,min=0.85,k=3 | 0.8740 | 0.4756 | 0.7959 |
| 12 | tokentriage-kstable | entropy=0.20,min=0.85,k=3 | 0.8711 | 0.4756 | 0.7959 |

## Interpretation

This experiment compares deployable TokenTriage policies against confidence-style early exits (entropy and margin), fixed-depth execution, and oracle stability upper bounds on real public radiology text. The paper-grade claim is the Pareto shape: policies that optimize confidence alone can retain high compute savings while leaving high disagreement on clinically critical tokens; TokenTriage variants spend extra layers selectively on high-risk tokens and reduce harm-weighted disagreement.

The current metric is final-model preservation, not radiologist correctness. That is appropriate for adaptive-inference evaluation, but a clinical paper should add RadGraph or radiologist-labeled entity correctness before making safety claims about patient outcomes.
