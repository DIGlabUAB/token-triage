# TokenTriage, Results

Harm-stratified evaluation of adaptive inference on clinical text. Confidence-only early exit
concentrates approximation **and ground-truth** error on clinically critical tokens; TokenTriage
removes most of it in the preservation/verification regime. All numbers regenerate from cached
traces (`scripts/run_paper_grade.py`, `aggregate_results.py`, `add_gold_correctness.py`,
`run_real_decoding.py`).

## 1. Main result, preservation (256 IU X-Ray reports, Qwen2.5-1.5B)

13,697 next-token positions · 1,163 critical tokens · 28 layers.

| Method | Setting | Compute saved | Critical disagreement | 95% CI | HWD |
| --- | --- | ---: | ---: | --- | ---: |
| entropy | threshold=0.10 | 0.379 | 0.717 | 0.686-0.747 | 0.560 |
| margin | threshold=0.50 | 0.571 | 0.865 | 0.848-0.884 | 0.779 |
| **tokentriage-kstable** | **ent=0.10,min=0.85,k=4** | **0.331** | **0.249** | **0.226-0.276** | **0.390** |
| tokentriage-kstable | ent=0.20,min=0.85,k=4 | 0.517 | 0.270 | 0.247-0.297 | 0.575 |

Entropy saves 37.9% compute but disagrees with the full model on 71.7% of critical tokens.
TokenTriage at the matched threshold cuts that to 24.9% (a 46.9-point reduction) at 33.1% savings.
Stable versus the 64-report run (0.721 to 0.255).

## 2. Generalization (preservation, critical-token disagreement vs full model)

| Run | Family | Layers | Critical tok | Baseline | TokenTriage | Compute saved |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B (64) | Qwen2.5 | 24 | 341 | 0.114 | 0.012 | 0.085 |
| Qwen2.5-1.5B (64) | Qwen2.5 | 28 | 341 | 0.721 | 0.255 | 0.329 |
| Qwen2.5-3B (64) | Qwen2.5 | 36 | 341 | 0.531 | 0.185 | 0.297 |
| SmolLM2-1.7B (64) | SmolLM2 | 24 | 349 | 0.891 | 0.307 | 0.830 |
| Qwen2.5-1.5B (256) | Qwen2.5 | 28 | 1,163 | 0.717 | 0.249 | 0.331 |
| ROCO chest / Qwen2.5-1.5B (64) | Qwen2.5 | 28 | 150 | 0.620 | 0.153 | 0.273 |

Two model families, a 6× size range, a second public dataset, and a 4× sample-size check, the
pattern holds everywhere. Smaller models show a smaller gap (when the baseline is already safe there
is little to fix), which indicates the method targets a real failure mode rather than inflating every
number. Phi-3.5-mini was attempted but excluded due to a `DynamicCache` incompatibility in its remote
modeling code.

## 3. Correctness, excess error vs the radiologist's actual words

Because the teacher-forced text is the radiologist's report, the gold next token is ground truth.
Exact-match of free prose has a high floor (valid synonyms), so the controlled quantity is the
**excess** critical-token error introduced over the full model.

| Model | Full-model floor | Entropy excess | TokenTriage excess |
| --- | ---: | ---: | ---: |
| Qwen2.5-0.5B | 0.771 | +0.021 | +0.000 |
| Qwen2.5-1.5B | 0.757 | +0.167 | +0.024 |
| Qwen2.5-3B | 0.733 | +0.173 | +0.029 |
| SmolLM2-1.7B | 0.782 | +0.215 | +0.020 |
| Qwen2.5-1.5B (256) | 0.742 | +0.190 | +0.025 |
| ROCO chest / Qwen2.5-1.5B | 0.693 | +0.187 | +0.040 |

Confidence-only exit adds 16-22 points of extra critical-token error against ground truth;
TokenTriage adds 0-4. No new labels required.

## 4. Entity-level correctness (CheXbert)

Token-level error need not be clinically meaningful. We measure clinical impact directly: apply each
policy's critical-token errors to the gold report and label gold vs perturbed reports with **CheXbert**
(14 CheXpert findings), then count how often the finding vector changes.

| Run | Entropy: reports changed | TokenTriage | Entropy: findings flipped | TokenTriage |
| --- | ---: | ---: | ---: | ---: |
| IU X-Ray (256) | 69.9% | **46.1%** | 400 | **224** |
| IU X-Ray (64) | 75.0% | **51.6%** | 99 | **67** |

Confidence-only exit's critical-token errors alter ≥1 CheXbert finding in 70% of reports (400 findings
flipped); TokenTriage reduces this to 46% (224 flipped), a **44% reduction in clinical-finding
changes**. It does not eliminate clinical change (the honest residual), but roughly halves it. This is
entity-level clinical correctness, not a token proxy. (Uses an isolated `.venv-chex`;
`build_chexbert_inputs.py` to `run_chexbert_eval.py`.)

## 5. Decoding regime, honest negative

Real free-running early-exit decoding (24 IU X-Ray continuations, Qwen2.5-1.5B, entropy 0.10),
emitting each token at its exit layer so errors propagate:

| Decoder | Critical-term fidelity to full | Prefix agreement | Compute saved |
| --- | ---: | ---: | ---: |
| entropy | 0.097 | 0.023 | 0.677 |
| tokentriage | 0.097 | 0.023 | 0.676 |

TokenTriage changed the emitted token in **0 of 7** critical interventions. At this operating point,
untrained logit-lens early exit collapses for *both* policies: when the model freely generates a
critical token it is already confident, so the cheap-exit token is stable and harm-gating does not
fire. This matches why deployed early-exit systems *train* exit heads (CALM, LayerSkip) rather than
read a raw logit lens. The contribution therefore lives in the **preservation / verification**
regime (teacher forcing, scoring, grounded generation, self-speculative-decoding acceptance), not in
naive untrained free generation. A trained harm-aware critical-token exit head is the natural next
step.

## Artifacts

- `outputs/pg-iuxray-qwen15b-256/`, main run (traces.jsonl, policy_results.csv, paper_results.md)
- `outputs/pg-iuxray-{qwen05b,qwen3b,smollm2}-64/`, `outputs/pg-roco-qwen15b-64/`, generalization
- `outputs/aggregate_summary.json`, `outputs/gold_correctness.json`, consolidated summaries
- `outputs/real-decoding-iuxray-qwen15b/`, decoding experiment (summary, records, overrides)
- `outputs/entity/entity_correctness.json`, CheXbert entity-level correctness
- `manuscript/main.pdf`, the paper · `project-page/`, the project page
