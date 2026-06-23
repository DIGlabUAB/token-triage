"""Step 1 of the entity-level correctness pass (runs in the experiment venv).

For each report, reconstruct three texts:
  - gold      : the radiologist's report (re-decoded from the gold ids)
  - entropy   : gold with every critical-token *error* of the entropy exit applied
  - triage    : gold with every critical-token *error* of TokenTriage applied
A critical-token "error" is a position where the policy's exit-layer top token differs
from the gold token. Only critical positions are perturbed, isolating critical-token harm.
The three texts are then labeled by CheXbert (step 2) to test whether the token errors a
policy makes actually flip a clinical finding.
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
ENT_DIR = OUT / "entity"

ENTROPY_THR = 0.10
MIN_FRAC = 0.85
PATIENCE = 4


def entropy_exit(entropy_by_layer, thr):
    for i, e in enumerate(entropy_by_layer, start=1):
        if e <= thr:
            return i
    return len(entropy_by_layer)


def fixed_exit(n, frac):
    return min(n, max(1, math.ceil(n * frac)))


def kstable_exit(top_ids, base, n, min_frac, k, crit):
    if crit < 0.8:
        return base
    start = max(base, fixed_exit(n, min_frac), k)
    for layer in range(start, n + 1):
        if len(set(top_ids[layer - k:layer])) == 1:
            return layer
    return n


def build(run_dir: Path):
    meta = json.loads((run_dir / "trace_meta.json").read_text())
    reports = json.loads((run_dir / "reports.json").read_text())
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(meta["model"], trust_remote_code=True)
    max_len = int(meta.get("max_length", 160))
    ids_by_row = {r["row_id"]: tok(r["text"], truncation=True, max_length=max_len,
                                   add_special_tokens=True)["input_ids"] for r in reports}

    traces = [json.loads(l) for l in (run_dir / "traces.jsonl").read_text().splitlines()]
    crit_by_row: dict[int, list[dict]] = {}
    for tr in traces:
        if tr["criticality"] >= 0.8:
            crit_by_row.setdefault(tr["row_id"], []).append(tr)

    out = []
    n_ent_changes = n_tri_changes = 0
    for r in reports:
        ids = ids_by_row[r["row_id"]]
        gold = list(ids)
        ent = list(ids)
        tri = list(ids)
        for tr in crit_by_row.get(r["row_id"], []):
            p = tr["position"]
            if p >= len(ids):
                continue
            gold_id = ids[p]
            top = tr["top_id_by_layer"]
            n = len(top)
            base = entropy_exit(tr["entropy_by_layer"], ENTROPY_THR)
            ke = kstable_exit(top, base, n, MIN_FRAC, PATIENCE, tr["criticality"])
            e_id = top[base - 1]
            t_id = top[ke - 1]
            if e_id != gold_id:
                ent[p] = e_id
                n_ent_changes += 1
            if t_id != gold_id:
                tri[p] = t_id
                n_tri_changes += 1
        out.append({
            "report_id": r["report_id"],
            "gold": tok.decode(gold, skip_special_tokens=True),
            "entropy": tok.decode(ent, skip_special_tokens=True),
            "triage": tok.decode(tri, skip_special_tokens=True),
        })
    ENT_DIR.mkdir(parents=True, exist_ok=True)
    name = run_dir.name
    (ENT_DIR / f"inputs_{name}.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"{name}: {len(out)} reports · {n_ent_changes} entropy critical edits · {n_tri_changes} triage critical edits")
    print(f"wrote {ENT_DIR / f'inputs_{name}.json'}")


def main():
    runs = sys.argv[1:] or ["pg-iuxray-qwen15b-256", "paper-grade-iuxray-qwen25-64-deployable"]
    for r in runs:
        d = OUT / r
        if (d / "traces.jsonl").exists():
            build(d)
        else:
            print(f"skip {r}: no traces")


if __name__ == "__main__":
    main()
