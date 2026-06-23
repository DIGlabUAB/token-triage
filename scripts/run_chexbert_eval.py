"""Steps 2-3 of the entity-level correctness pass (runs in the CheXbert venv: .venv-chex).

Labels gold / entropy-perturbed / triage-perturbed reports with CheXbert (14 CheXpert
findings) and measures how often each policy's critical-token errors change the clinical
label vector relative to the gold report. This is clinical-entity correctness: do the token
errors actually flip a finding?
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parents[1]
ENT_DIR = ROOT / "outputs" / "entity"


def hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def main():
    from f1chexbert import F1CheXbert
    chex = F1CheXbert()
    labels14 = chex.target_names

    inputs = sorted(ENT_DIR.glob("inputs_*.json"))
    summary = []
    for path in inputs:
        run = path.stem.replace("inputs_", "")
        reports = json.loads(path.read_text())
        cache = {}

        def label(text):
            if text not in cache:
                cache[text] = chex.get_label(text)
            return cache[text]

        rows = []
        agg = {
            "n": 0, "ent_any": 0, "tri_any": 0,
            "ent_ham": 0, "tri_ham": 0,
            "ent_findings_added": 0, "ent_findings_dropped": 0,
            "tri_findings_added": 0, "tri_findings_dropped": 0,
        }
        for i, r in enumerate(reports):
            g = label(r["gold"]); e = label(r["entropy"]); t = label(r["triage"])
            eh, th = hamming(g, e), hamming(g, t)
            agg["n"] += 1
            agg["ent_any"] += int(eh > 0)
            agg["tri_any"] += int(th > 0)
            agg["ent_ham"] += eh
            agg["tri_ham"] += th
            agg["ent_findings_added"] += sum(1 for gi, ei in zip(g, e) if ei and not gi)
            agg["ent_findings_dropped"] += sum(1 for gi, ei in zip(g, e) if gi and not ei)
            agg["tri_findings_added"] += sum(1 for gi, ti in zip(g, t) if ti and not gi)
            agg["tri_findings_dropped"] += sum(1 for gi, ti in zip(g, t) if gi and not ti)
            rows.append({"report_id": r["report_id"], "gold": g, "entropy": e, "triage": t,
                         "entropy_changed": eh > 0, "triage_changed": th > 0})
            if (i + 1) % 32 == 0:
                print(f"  {run}: {i+1}/{len(reports)}", flush=True)
        (ENT_DIR / f"labels_{run}.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        n = agg["n"]
        s = {
            "run": run, "labels": labels14, "reports": n,
            "entropy_report_change_rate": round(agg["ent_any"] / n, 4),
            "triage_report_change_rate": round(agg["tri_any"] / n, 4),
            "entropy_finding_discordance": round(agg["ent_ham"] / (n * 14), 4),
            "triage_finding_discordance": round(agg["tri_ham"] / (n * 14), 4),
            "entropy_findings_flipped": agg["ent_ham"],
            "triage_findings_flipped": agg["tri_ham"],
            "entropy_added/dropped": [agg["ent_findings_added"], agg["ent_findings_dropped"]],
            "triage_added/dropped": [agg["tri_findings_added"], agg["tri_findings_dropped"]],
        }
        summary.append(s)
        print(f"\n{run}: reports={n}")
        print(f"  CheXbert label CHANGED:   entropy {s['entropy_report_change_rate']*100:.1f}% of reports | "
              f"triage {s['triage_report_change_rate']*100:.1f}%")
        print(f"  findings flipped (total): entropy {agg['ent_ham']} | triage {agg['tri_ham']}")

    (ENT_DIR / "entity_correctness.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {ENT_DIR/'entity_correctness.json'}")


if __name__ == "__main__":
    main()
