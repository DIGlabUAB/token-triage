# TokenTriage

**Harm-aware token routing for clinical language model inference.** A harm-stratified evaluation
lens showing that confidence-only early exit concentrates its approximation and ground-truth error
on clinically critical tokens, and a simple routing rule (TokenTriage) that removes most of it.

- 📄 **Paper:** [`manuscript/main.pdf`](manuscript/main.pdf) · 🌐 **Project page:** [`project-page/`](project-page/index.html) · 📊 **Results:** [`RESULTS.md`](RESULTS.md)

### What it shows (validated across 4 open models, 2 datasets, and against radiologist text)

- Confidence-only early exit disagrees with the full model on **71.7%** of clinically critical tokens (256 IU X-Ray reports, Qwen2.5-1.5B); TokenTriage cuts that to **24.9%** at comparable compute.
- The pattern holds across **Qwen2.5-0.5B/1.5B/3B** and **SmolLM2-1.7B**, and on a second dataset (ROCO chest).
- **Against ground truth:** confidence-only exit adds **16-22 points** of extra critical-token error vs the radiologist's actual words; TokenTriage adds **0-4**.
- **Honest negative:** naive untrained free-running early exit collapses for all policies, the contribution lives in the preservation/verification regime (e.g. self-speculative decoding).

### Pipeline

- Teacher-forced logit-lens traces from open HF models on public radiology text.
- Pluggable clinical risk tagger (negation, laterality, severity, temporality, medication, measurement, units, findings).
- Confidence-only early exit vs harm-aware TokenTriage routing, scored on identical cached traces.
- Reproducible JSON / CSV / Markdown artifacts under `outputs/` with bootstrap CIs.

### Key scripts

```bash
python scripts/run_paper_grade.py --limit 256 --output-dir outputs/pg-iuxray-qwen15b-256  # main run
python scripts/aggregate_results.py        # preservation summary across runs
python scripts/add_gold_correctness.py     # excess error vs gold report text
python scripts/run_real_decoding.py        # real free-running early-exit decoding
python scripts/create_model_figures.py     # cross-model + correctness figures
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --no-use-pep517 -e .
python -m token_triage.cli --model qwen2.5:1.5b --cases 8 --output-dir outputs/run
```

The DOCX requested `qwen2.5:1.5b`, and the end-to-end verification run used that local Ollama model. If you need to reproduce setup from a clean machine, pull it first:

```bash
ollama pull qwen2.5:1.5b
python -m token_triage.cli --model qwen2.5:1.5b
```

## Outputs

- `summary.json`: aggregate comparison between early-exit strategies.
- `cases.json`: generated or templated synthetic radiology cases.
- `token_records.csv`: token-level criticality, confidence proxy, and simulated exit layers.
- `report.md`: concise experiment report.

## Important Limitation

Ollama was verified locally, but the installed endpoint did not return generated-token `logprobs` even when requested through `/v1/completions`. This first end-to-end build therefore uses a deterministic `confidence_proxy` and isolates the scorer boundary so true logprobs or hidden-state/logit-lens scores can replace it later.

The current claim should be framed carefully: early exit is not new; the underexplored piece is **clinical safety-aware token triage**.

## Real HF Logit-Lens Experiment

This runner uses public Hugging Face radiology text plus hidden states from an open Qwen model:

```bash
python scripts/run_hf_logit_lens.py \
  --dataset ChayanM/IUXray-Data-Train-Test \
  --config default \
  --split test \
  --limit 32 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir outputs/hf-logit-lens
```

It writes:

- `summary.json`: compute/safety comparison.
- `token_records.csv`: per-token exit layers, criticality, NLL, and final-layer disagreement.
- `report.md`: concise interpretation.

## Paper-Grade Baseline Suite

The SOTA-style runner caches per-layer traces once, then evaluates confidence baselines, fixed-depth baselines, oracle upper bounds, and deployable TokenTriage variants:

```bash
python scripts/run_paper_grade.py \
  --limit 64 \
  --max-length 160 \
  --bootstrap-samples 200 \
  --output-dir outputs/paper-grade-iuxray-qwen25-64-deployable
```

Primary outputs:

- `traces.jsonl`: reusable per-token per-layer logit-lens traces.
- `policy_results.csv`: all policy/threshold results.
- `paper_results.md`: paper-style comparison table.

The current headline result is in `RESULTS.md`.
