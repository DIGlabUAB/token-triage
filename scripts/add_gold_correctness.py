"""Correctness-vs-gold analysis.

The teacher-forced text IS the radiologist's report, so the gold next-token is ground
truth. For each policy we measure how many EXTRA critical-token errors-vs-gold it
introduces over the full model. This turns the preservation metric into a correctness
metric with no new labels. Gold token ids are recovered by re-tokenizing (no GPU re-run).
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

RUNS = [
    ("Qwen2.5-0.5B", "Qwen2.5", "pg-iuxray-qwen05b-64"),
    ("Qwen2.5-1.5B", "Qwen2.5", "paper-grade-iuxray-qwen25-64-deployable"),
    ("Qwen2.5-3B", "Qwen2.5", "pg-iuxray-qwen3b-64"),
    ("SmolLM2-1.7B", "SmolLM2", "pg-iuxray-smollm2-64"),
    ("Phi-3.5-mini", "Phi-3.5", "pg-iuxray-phi35-64"),
    ("Qwen2.5-1.5B (256)", "Qwen2.5", "pg-iuxray-qwen15b-256"),
    ("ROCO chest / Qwen2.5-1.5B", "Qwen2.5", "pg-roco-qwen15b-64"),
]

ENTROPY_THR = 0.10
MIN_FRAC = 0.85
PATIENCE = 4


def entropy_exit(entropy_by_layer, threshold):
    for idx, e in enumerate(entropy_by_layer, start=1):
        if e <= threshold:
            return idx
    return len(entropy_by_layer)


def fixed_exit(n, frac):
    return min(n, max(1, math.ceil(n * frac)))


def kstable_exit(top_ids, base_exit, n, min_frac, k, criticality):
    if criticality < 0.8:
        return base_exit
    start = max(base_exit, fixed_exit(n, min_frac), k)
    for layer in range(start, n + 1):
        if len(set(top_ids[layer - k:layer])) == 1:
            return layer
    return n


def analyze(label, family, d):
    run_dir = OUT / d
    if not (run_dir / "traces.jsonl").exists():
        return None
    meta = json.loads((run_dir / "trace_meta.json").read_text())
    reports = json.loads((run_dir / "reports.json").read_text())
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(meta["model"], trust_remote_code=True)
    max_len = int(meta.get("max_length", 160))
    ids_by_row = {}
    for r in reports:
        enc = tok(r["text"], truncation=True, max_length=max_len, add_special_tokens=True)
        ids_by_row[r["row_id"]] = enc["input_ids"]

    traces = [json.loads(l) for l in (run_dir / "traces.jsonl").read_text().splitlines()]
    # accumulators over critical tokens
    crit = {"n": 0, "full_err": 0, "ent_err": 0, "tri_err": 0, "aligned": 0}
    for tr in traces:
        if tr["criticality"] < 0.8:
            continue
        ids = ids_by_row.get(tr["row_id"])
        if ids is None or tr["position"] >= len(ids):
            continue
        target_id = ids[tr["position"]]
        # sanity: decoded gold should match stored token text (allow whitespace diffs)
        if tok.decode([target_id]).strip() != tr["token_text"].strip():
            continue
        crit["aligned"] += 1
        crit["n"] += 1
        top = tr["top_id_by_layer"]
        n = len(top)
        base = entropy_exit(tr["entropy_by_layer"], ENTROPY_THR)
        tri = kstable_exit(top, base, n, MIN_FRAC, PATIENCE, tr["criticality"])
        crit["full_err"] += int(top[n - 1] != target_id)
        crit["ent_err"] += int(top[base - 1] != target_id)
        crit["tri_err"] += int(top[tri - 1] != target_id)

    n = crit["n"] or 1
    full = crit["full_err"] / n
    ent = crit["ent_err"] / n
    tri = crit["tri_err"] / n
    return {
        "label": label, "family": family, "dir": d,
        "model": meta["model"], "reports": meta.get("limit"),
        "critical_aligned": crit["aligned"],
        "full_gold_error": round(full, 4),
        "entropy_gold_error": round(ent, 4),
        "triage_gold_error": round(tri, 4),
        "entropy_excess_error": round(ent - full, 4),
        "triage_excess_error": round(tri - full, 4),
        "excess_reduction": round((ent - full) - (tri - full), 4),
    }


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    results = []
    for label, family, d in RUNS:
        if only and only not in d:
            continue
        try:
            r = analyze(label, family, d)
        except Exception as e:  # noqa: BLE001
            print(f"skip {d}: {type(e).__name__}: {e}", file=sys.stderr)
            r = None
        if r:
            results.append(r)
    if not only:
        (OUT / "gold_correctness.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"{'run':<28}{'critN':>6}{'full':>7}{'ent':>7}{'tri':>7}  {'entXS':>7}{'triXS':>7}{'red':>7}")
    for r in results:
        print(f"{r['label']:<28}{r['critical_aligned']:>6}"
              f"{r['full_gold_error']*100:>6.1f} {r['entropy_gold_error']*100:>6.1f} {r['triage_gold_error']*100:>6.1f}"
              f"  {r['entropy_excess_error']*100:>6.1f} {r['triage_excess_error']*100:>6.1f} {r['excess_reduction']*100:>6.1f}")
    print("\nXS = excess error-vs-gold introduced over full model; red = reduction in excess (entXS - triXS)")


if __name__ == "__main__":
    main()
