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
| tokentriage-kstable | entropy=0.10,min=0.85,k=4 | 0.3293 | 0.2551 | 0.3902 | 5.1208 | 0.2143-0.2933 |
| tokentriage-kstable | entropy=0.20,min=0.85,k=4 | 0.5130 | 0.2874 | 0.5779 | 7.3492 | 0.2473-0.3333 |
| tokentriage-stable | entropy=0.55,min=0.50 | 0.8680 | 0.0000 | 0.6157 | 10.3903 | 0.0000-0.0000 |

## Top Deployable Pareto Candidates

| Rank | Method | Setting | Compute saved | Critical disagreement | HWD |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | fixed-depth | 1.00 | 0.0000 | 0.0000 | 0.0000 |
| 2 | tokentriage-kstable | entropy=0.10,min=0.85,k=4 | 0.3293 | 0.2551 | 0.3902 |
| 3 | tokentriage-kstable | entropy=0.15,min=0.85,k=4 | 0.4377 | 0.2727 | 0.5053 |
| 4 | tokentriage-kstable | entropy=0.20,min=0.85,k=4 | 0.5130 | 0.2874 | 0.5779 |
| 5 | tokentriage-kstable | entropy=0.42,min=0.85,k=4 | 0.7738 | 0.2933 | 0.7075 |
| 6 | tokentriage-kstable | entropy=0.30,min=0.85,k=4 | 0.6386 | 0.2933 | 0.6636 |
| 7 | tokentriage-kstable | entropy=0.55,min=0.85,k=4 | 0.8678 | 0.2991 | 0.7241 |
| 8 | tokentriage-kstable | entropy=0.10,min=0.72,k=4 | 0.3335 | 0.3490 | 0.4247 |
| 9 | tokentriage-kstable | entropy=0.10,min=0.85,k=3 | 0.3307 | 0.3578 | 0.4281 |
| 10 | tokentriage-kstable | entropy=0.15,min=0.72,k=4 | 0.4420 | 0.3724 | 0.5420 |
| 11 | tokentriage-kstable | entropy=0.20,min=0.72,k=4 | 0.5174 | 0.3900 | 0.6156 |
| 12 | tokentriage-kstable | entropy=0.15,min=0.85,k=3 | 0.4393 | 0.3930 | 0.5495 |

## Interpretation

This experiment compares deployable TokenTriage policies against confidence-style early exits (entropy and margin), fixed-depth execution, and oracle stability upper bounds on real public radiology text. The paper-grade claim is the Pareto shape: policies that optimize confidence alone can retain high compute savings while leaving high disagreement on clinically critical tokens; TokenTriage variants spend extra layers selectively on high-risk tokens and reduce harm-weighted disagreement.

The current metric is final-model preservation, not radiologist correctness. That is appropriate for adaptive-inference evaluation, but a clinical paper should add RadGraph or radiologist-labeled entity correctness before making safety claims about patient outcomes.
