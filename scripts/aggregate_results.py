"""Aggregate paper-grade runs into one machine-readable summary + markdown table.

Reads every run directory's policy_results.csv and trace_meta.json, then for each run
computes the strongest confidence baseline and the matched deployable TokenTriage point.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

# (label, family, output_dir, role)
RUNS = [
    ("Qwen2.5-0.5B", "Qwen2.5", "pg-iuxray-qwen05b-64", "family"),
    ("Qwen2.5-1.5B", "Qwen2.5", "paper-grade-iuxray-qwen25-64-deployable", "family"),
    ("Qwen2.5-3B", "Qwen2.5", "pg-iuxray-qwen3b-64", "family"),
    ("SmolLM2-1.7B", "SmolLM2", "pg-iuxray-smollm2-64", "family"),
    ("Phi-3.5-mini", "Phi-3.5", "pg-iuxray-phi35-64", "family"),
    ("Qwen2.5-1.5B (256)", "Qwen2.5", "pg-iuxray-qwen15b-256", "scale"),
    ("ROCO chest / Qwen2.5-1.5B", "Qwen2.5", "pg-roco-qwen15b-64", "dataset"),
]


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def f(row: dict, key: str) -> float:
    return float(row[key])


def best_by_crit(rows: list[dict], policy: str) -> dict | None:
    cands = [r for r in rows if r["policy"] == policy]
    return min(cands, key=lambda r: f(r, "disagreement_critical")) if cands else None


def matched_triage(rows: list[dict], ent_threshold: float) -> dict | None:
    target = f"entropy={ent_threshold:.2f},min=0.85,k=4"
    exact = [r for r in rows if r["policy"] == "tokentriage-kstable" and r["setting"] == target]
    if exact:
        return exact[0]
    cands = [r for r in rows if r["policy"] in {"tokentriage-kstable", "tokentriage-minlayer"}]
    return min(cands, key=lambda r: (f(r, "disagreement_critical"), -f(r, "compute_saved"))) if cands else None


def summarize() -> list[dict]:
    out = []
    for label, family, d, role in RUNS:
        run_dir = OUT / d
        csv_path = run_dir / "policy_results.csv"
        meta_path = run_dir / "trace_meta.json"
        if not csv_path.exists():
            out.append({"label": label, "dir": d, "status": "missing"})
            continue
        rows = read_rows(csv_path)
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        be = best_by_crit(rows, "entropy")
        bm = best_by_crit(rows, "margin")
        baseline = min([r for r in (be, bm) if r], key=lambda r: f(r, "disagreement_critical"))
        ent_thr = float(be["setting"].split("=")[1]) if be else 0.10
        tri = matched_triage(rows, ent_thr)
        out.append({
            "label": label,
            "family": family,
            "role": role,
            "dir": d,
            "status": "ok",
            "model": meta.get("model"),
            "dataset": meta.get("dataset"),
            "reports": meta.get("limit"),
            "layers": meta.get("layers"),
            "tokens": int(tri["tokens"]) if tri else None,
            "critical_tokens": int(tri["critical_tokens"]) if tri else None,
            "baseline_policy": baseline["policy"],
            "baseline_setting": baseline["setting"],
            "baseline_crit": round(f(baseline, "disagreement_critical"), 4),
            "baseline_compute": round(f(baseline, "compute_saved"), 4),
            "entropy_crit": round(f(be, "disagreement_critical"), 4) if be else None,
            "entropy_compute": round(f(be, "compute_saved"), 4) if be else None,
            "triage_setting": tri["setting"] if tri else None,
            "triage_crit": round(f(tri, "disagreement_critical"), 4) if tri else None,
            "triage_compute": round(f(tri, "compute_saved"), 4) if tri else None,
            "triage_hwd": round(f(tri, "harm_weighted_disagreement"), 4) if tri else None,
            "triage_ci": tri.get("disagreement_critical_ci95") if tri else None,
        })
    return out


def main() -> None:
    summary = [s for s in summarize() if s.get("status") == "ok"]
    (OUT / "aggregate_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"{'run':<28}{'rep':>4}{'tok':>6}{'crit':>5}  {'base%':>7}{'tri%':>7}{'save%':>7}")
    for s in summary:
        print(f"{s['label']:<28}{s['reports']!s:>4}{s['tokens']!s:>6}{s['critical_tokens']!s:>5}"
              f"  {s['baseline_crit']*100:>6.1f} {s['triage_crit']*100:>6.1f} {s['triage_compute']*100:>6.1f}")
    print(f"\nWrote {OUT/'aggregate_summary.json'} ({len(summary)} runs)")


if __name__ == "__main__":
    main()
