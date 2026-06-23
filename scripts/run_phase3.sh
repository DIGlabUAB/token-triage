#!/bin/bash
set -u
cd /Users/boddu/Documents/govinda/codex/TokenTriage
export TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python
LOG=outputs/phase3_progress.log
echo "=== phase3 waiting for retry batch to finish $(date) ===" > "$LOG"
# wait until the retry batch (256 + phi) is done to avoid MPS contention
until grep -q "=== retry finished" outputs/retry_progress.log 2>/dev/null; do sleep 15; done
echo "=== phase3 started $(date) ===" >> "$LOG"
echo ">>> roco-64 (corrected cols) @ $(date)" | tee -a "$LOG"
$PY scripts/run_paper_grade.py --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset Santhosh1705kumar/radiology-reports-chest --split test \
  --text-column caption --id-column image_id \
  --limit 64 --max-length 120 --bootstrap-samples 200 \
  --output-dir outputs/pg-roco-qwen15b-64 >> "$LOG" 2>&1
echo "<<< roco-64 done rc=$? @ $(date)" | tee -a "$LOG"
echo ">>> real-decoding @ $(date)" | tee -a "$LOG"
$PY scripts/run_real_decoding.py --model Qwen/Qwen2.5-1.5B-Instruct \
  --limit 24 --prompt-tokens 8 --max-new 50 \
  --output-dir outputs/real-decoding-iuxray-qwen15b >> "$LOG" 2>&1
echo "<<< real-decoding done rc=$? @ $(date)" | tee -a "$LOG"
echo "=== phase3 finished $(date) ===" | tee -a "$LOG"
