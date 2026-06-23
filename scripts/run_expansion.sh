#!/bin/bash
set -u
cd /Users/boddu/Documents/govinda/codex/TokenTriage
export TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python
LOG=outputs/expansion_progress.log
echo "=== expansion started $(date) ===" > "$LOG"

run () {
  local name="$1"; shift
  echo ">>> $name @ $(date)" | tee -a "$LOG"
  $PY scripts/run_paper_grade.py "$@" >> "$LOG" 2>&1
  echo "<<< $name done rc=$? @ $(date)" | tee -a "$LOG"
}

# within-family scaling ladder on IU X-Ray @ 64 (1.5B already exists)
run qwen05b-64  --model Qwen/Qwen2.5-0.5B-Instruct --limit 64 --bootstrap-samples 200 --output-dir outputs/pg-iuxray-qwen05b-64
run smollm2-64  --model HuggingFaceTB/SmolLM2-1.7B-Instruct --limit 64 --bootstrap-samples 200 --output-dir outputs/pg-iuxray-smollm2-64
run qwen3b-64   --model Qwen/Qwen2.5-3B-Instruct --limit 64 --bootstrap-samples 200 --output-dir outputs/pg-iuxray-qwen3b-64
# second dataset at scale
run roco-64     --model Qwen/Qwen2.5-1.5B-Instruct --dataset Santhosh1705kumar/radiology-reports-chest --limit 64 --bootstrap-samples 200 --output-dir outputs/pg-roco-qwen15b-64
# headline scale-up
run qwen15b-256 --model Qwen/Qwen2.5-1.5B-Instruct --limit 256 --bootstrap-samples 200 --output-dir outputs/pg-iuxray-qwen15b-256
# extra family (largest/slowest, last)
run phi35-64    --model microsoft/Phi-3.5-mini-instruct --limit 64 --bootstrap-samples 200 --output-dir outputs/pg-iuxray-phi35-64
echo "=== expansion finished $(date) ===" | tee -a "$LOG"
