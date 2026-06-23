"""Render the cross-model-family generalization figure from aggregate_summary.json."""
from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "outputs" / "aggregate_summary.json"
GOLD = ROOT / "outputs" / "gold_correctness.json"
ENTITY = ROOT / "outputs" / "entity" / "entity_correctness.json"
ASSET = ROOT / "project-page" / "assets"

C = {
    "ink": "#14201c", "muted": "#707a72", "line": "#e2ddd0",
    "paper": "#fffdf8", "emerald": "#0e6f63", "emerald_deep": "#0a4f47",
    "clay": "#b1493b", "wash": "#efece2",
}


def esc(v: object) -> str:
    return html.escape(str(v), quote=True)


def t(x, y, s, size=14, w=500, fill="ink", anchor="start", mono=True):
    fam = "IBM Plex Mono, ui-monospace, monospace" if mono else "JetBrains Mono, IBM Plex Mono, ui-monospace, monospace"
    return (f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" '
            f'font-weight="{w}" fill="{C[fill]}" text-anchor="{anchor}">{esc(s)}</text>')


def serif(x, y, s, size=26, w=460, fill="ink"):
    return t(x, y, s, size, w, fill, "start", mono=False)


def svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img">'
            f'<rect width="{w}" height="{h}" fill="{C["paper"]}"/>{body}</svg>\n')


def make_models() -> None:
    data = json.loads(SUMMARY.read_text())
    rows = [s for s in data if s.get("role") == "family"]
    # order small -> large by family then label
    order = {"Qwen2.5-0.5B": 0, "SmolLM2-1.7B": 1, "Qwen2.5-1.5B": 2, "Qwen2.5-3B": 3, "Phi-3.5-mini": 4}
    rows.sort(key=lambda s: order.get(s["label"], 9))
    if not rows:
        return

    W, left, bar_x, bar_w = 900, 60, 320, 430
    row_h = 96
    top = 132
    H = top + row_h * len(rows) + 30
    parts = [
        serif(46, 50, "Generalization across model families", 28),
        t(48, 80, "critical-token disagreement: best confidence baseline vs TokenTriage  ·  IU X-Ray, 64 reports", 13, 500, "muted"),
        # legend
        f'<rect x="{bar_x}" y="100" width="12" height="12" rx="2" fill="{C["clay"]}"/>',
        t(bar_x + 18, 110, "confidence baseline", 12, 600, "muted"),
        f'<rect x="{bar_x+170}" y="100" width="12" height="12" rx="2" fill="{C["emerald"]}"/>',
        t(bar_x + 188, 110, "TokenTriage", 12, 600, "muted"),
    ]
    y = top
    for s in rows:
        base = s["baseline_crit"]
        tri = s["triage_crit"]
        save = s["triage_compute"]
        parts.append(t(left, y + 16, s["label"], 15, 700, "ink"))
        parts.append(t(left, y + 38, f'{s["family"]} · {s["layers"]} layers · {s["critical_tokens"]} crit. tok', 11, 500, "muted"))
        # baseline bar
        parts.append(f'<rect x="{bar_x}" y="{y}" width="{bar_w}" height="22" rx="5" fill="{C["wash"]}" stroke="{C["line"]}"/>')
        parts.append(f'<rect x="{bar_x}" y="{y}" width="{bar_w*base:.1f}" height="22" rx="5" fill="{C["clay"]}"/>')
        parts.append(t(bar_x + bar_w + 12, y + 17, f'{base*100:.1f}%', 13, 700, "clay"))
        # triage bar
        parts.append(f'<rect x="{bar_x}" y="{y+30}" width="{bar_w}" height="22" rx="5" fill="{C["wash"]}" stroke="{C["line"]}"/>')
        parts.append(f'<rect x="{bar_x}" y="{y+30}" width="{bar_w*tri:.1f}" height="22" rx="5" fill="{C["emerald"]}"/>')
        parts.append(t(bar_x + bar_w + 12, y + 47, f'{tri*100:.1f}%', 13, 700, "emerald_deep"))
        parts.append(t(left, y + 62, f'TokenTriage keeps {save*100:.0f}% compute saved', 11, 500, "muted"))
        y += row_h
    ASSET.mkdir(parents=True, exist_ok=True)
    (ASSET / "models.svg").write_text(svg(W, H, "\n".join(parts)), encoding="utf-8")
    print(f"Wrote {ASSET/'models.svg'} ({len(rows)} models)")


def make_correctness() -> None:
    if not GOLD.exists():
        return
    data = json.loads(GOLD.read_text())
    rows = [r for r in data if "256" not in r["label"] and "ROCO" not in r["label"]]
    order = {"Qwen2.5-0.5B": 0, "SmolLM2-1.7B": 1, "Qwen2.5-1.5B": 2, "Qwen2.5-3B": 3, "Phi-3.5-mini": 4}
    rows.sort(key=lambda r: order.get(r["label"], 9))
    if not rows:
        return
    W, left, bar_x, bar_w = 900, 60, 360, 360
    row_h, top = 84, 150
    H = top + row_h * len(rows) + 30
    maxv = max(max(r["entropy_excess_error"], r["triage_excess_error"]) for r in rows) or 0.01
    parts = [
        serif(46, 50, "Extra errors against the radiologist's own words", 26),
        t(48, 80, "excess critical-token error introduced over the full model  ·  lower is safer  ·  IU X-Ray, 64 reports", 13, 500, "muted"),
        t(48, 104, "(this is correctness vs ground-truth report text, not self-consistency)", 12, 500, "muted"),
        f'<rect x="{bar_x}" y="120" width="12" height="12" rx="2" fill="{C["clay"]}"/>',
        t(bar_x + 18, 130, "confidence baseline", 12, 600, "muted"),
        f'<rect x="{bar_x+170}" y="120" width="12" height="12" rx="2" fill="{C["emerald"]}"/>',
        t(bar_x + 188, 130, "TokenTriage", 12, 600, "muted"),
    ]
    y = top
    for r in rows:
        e, tr = r["entropy_excess_error"], r["triage_excess_error"]
        parts.append(t(left, y + 18, r["label"], 15, 700, "ink"))
        parts.append(t(left, y + 40, f'{r["family"]} · full-model floor {r["full_gold_error"]*100:.0f}%', 11, 500, "muted"))
        parts.append(f'<rect x="{bar_x}" y="{y+4}" width="{bar_w}" height="20" rx="5" fill="{C["wash"]}" stroke="{C["line"]}"/>')
        parts.append(f'<rect x="{bar_x}" y="{y+4}" width="{bar_w*e/maxv:.1f}" height="20" rx="5" fill="{C["clay"]}"/>')
        parts.append(t(bar_x + bar_w + 12, y + 20, f'+{e*100:.1f} pt', 13, 700, "clay"))
        parts.append(f'<rect x="{bar_x}" y="{y+30}" width="{bar_w}" height="20" rx="5" fill="{C["wash"]}" stroke="{C["line"]}"/>')
        parts.append(f'<rect x="{bar_x}" y="{y+30}" width="{bar_w*tr/maxv:.1f}" height="20" rx="5" fill="{C["emerald"]}"/>')
        parts.append(t(bar_x + bar_w + 12, y + 46, f'+{tr*100:.1f} pt', 13, 700, "emerald_deep"))
        y += row_h
    ASSET.mkdir(parents=True, exist_ok=True)
    (ASSET / "correctness.svg").write_text(svg(W, H, "\n".join(parts)), encoding="utf-8")
    print(f"Wrote {ASSET/'correctness.svg'} ({len(rows)} models)")


def make_entity() -> None:
    if not ENTITY.exists():
        return
    data = json.loads(ENTITY.read_text())
    label_of = {"pg-iuxray-qwen15b-256": "IU X-Ray · 256 reports",
                "paper-grade-iuxray-qwen25-64-deployable": "IU X-Ray · 64 reports"}
    rows = [r for r in data if r["run"] in label_of]
    rows.sort(key=lambda r: 0 if "256" in r["run"] else 1)
    if not rows:
        return
    W, left, bar_x, bar_w = 900, 60, 330, 400
    row_h, top = 104, 160
    H = top + row_h * len(rows) + 20
    parts = [
        serif(46, 50, "Do the token errors change the diagnosis?", 26),
        t(48, 80, "share of reports whose CheXbert findings change when a policy's critical-token errors are applied", 13, 500, "muted"),
        t(48, 102, "Qwen2.5-1.5B · 14 CheXpert findings · lower is safer", 12, 500, "muted"),
        f'<rect x="{bar_x}" y="120" width="12" height="12" rx="2" fill="{C["clay"]}"/>',
        t(bar_x + 18, 130, "confidence baseline", 12, 600, "muted"),
        f'<rect x="{bar_x+170}" y="120" width="12" height="12" rx="2" fill="{C["emerald"]}"/>',
        t(bar_x + 188, 130, "TokenTriage", 12, 600, "muted"),
    ]
    y = top
    for r in rows:
        e = r["entropy_report_change_rate"]; tr = r["triage_report_change_rate"]
        ef = r["entropy_findings_flipped"]; tf = r["triage_findings_flipped"]
        parts.append(t(left, y + 16, label_of[r["run"]], 15, 700, "ink"))
        parts.append(t(left, y + 38, f'{ef} to {tf} total findings flipped', 11, 500, "muted"))
        parts.append(f'<rect x="{bar_x}" y="{y}" width="{bar_w}" height="22" rx="5" fill="{C["wash"]}" stroke="{C["line"]}"/>')
        parts.append(f'<rect x="{bar_x}" y="{y}" width="{bar_w*e:.1f}" height="22" rx="5" fill="{C["clay"]}"/>')
        parts.append(t(bar_x + bar_w + 12, y + 17, f'{e*100:.1f}%', 13, 700, "clay"))
        parts.append(f'<rect x="{bar_x}" y="{y+30}" width="{bar_w}" height="22" rx="5" fill="{C["wash"]}" stroke="{C["line"]}"/>')
        parts.append(f'<rect x="{bar_x}" y="{y+30}" width="{bar_w*tr:.1f}" height="22" rx="5" fill="{C["emerald"]}"/>')
        parts.append(t(bar_x + bar_w + 12, y + 47, f'{tr*100:.1f}%', 13, 700, "emerald_deep"))
        y += row_h
    ASSET.mkdir(parents=True, exist_ok=True)
    (ASSET / "entity.svg").write_text(svg(W, H, "\n".join(parts)), encoding="utf-8")
    print(f"Wrote {ASSET/'entity.svg'} ({len(rows)} runs)")


if __name__ == "__main__":
    make_models()
    make_correctness()
    make_entity()
