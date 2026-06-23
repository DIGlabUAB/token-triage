#!/bin/bash
set -u
cd /Users/boddu/Documents/govinda/codex/TokenTriage
export TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python
LOG=outputs/retry_progress.log
echo "=== retry started $(date) ===" > "$LOG"
run () { local n="$1"; shift; echo ">>> $n @ $(date)" | tee -a "$LOG"; $PY scripts/run_paper_grade.py "$@" >> "$LOG" 2>&1; echo "<<< $n done rc=$? @ $(date)" | tee -a "$LOG"; sleep 20; }
run roco-64     --model Qwen/Qwen2.5-1.5B-Instruct --dataset Santhosh1705kumar/radiology-reports-chest --limit 64 --bootstrap-samples 200 --output-dir outputs/pg-roco-qwen15b-64
run qwen15b-256 --model Qwen/Qwen2.5-1.5B-Instruct --limit 256 --bootstrap-samples 200 --output-dir outputs/pg-iuxray-qwen15b-256
run phi35-64    --model microsoft/Phi-3.5-mini-instruct --limit 64 --bootstrap-samples 200 --output-dir outputs/pg-iuxray-phi35-64
echo "=== retry finished $(date) ===" | tee -a "$LOG"
