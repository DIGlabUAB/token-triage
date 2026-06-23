from __future__ import annotations

import csv
import html
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "outputs" / "paper-grade-iuxray-qwen25-64-deployable"
EXTENDED_RUNS = [
    (
        "IU X-Ray / Qwen2.5-1.5B",
        ROOT / "outputs" / "paper-grade-iuxray-qwen25-64-deployable",
        "entropy=0.10,min=0.85,k=4",
    ),
    (
        "ROCO chest / Qwen2.5-1.5B",
        ROOT / "outputs" / "paper-grade-roco-chest-qwen25-16",
        None,
    ),
    (
        "IU X-Ray / Qwen2.5-0.5B",
        ROOT / "outputs" / "paper-grade-iuxray-qwen25-05b-16",
        None,
    ),
]
ASSET_DIR = ROOT / "project-page" / "assets"
LAYERS = 28


COLORS = {
    "ink": "#172023",
    "muted": "#687174",
    "line": "#d8d8d0",
    "paper": "#ffffff",
    "wash": "#f6f6f1",
    "teal": "#0f7f7b",
    "red": "#b84f42",
    "amber": "#c88a2d",
    "blue": "#3f5870",
    "green": "#607d5d",
}


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def read_policy_rows() -> list[dict[str, str]]:
    with (RUN_DIR / "policy_results.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_policy_rows_from(path: Path) -> list[dict[str, str]]:
    with (path / "policy_results.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_meta(path: Path) -> dict:
    meta_path = path / "trace_meta.json"
    return json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}


def read_traces() -> list[dict]:
    with (RUN_DIR / "traces.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def entropy_exit(trace: dict, threshold: float) -> int:
    for idx, entropy in enumerate(trace["entropy_by_layer"], start=1):
        if entropy <= threshold:
            return idx
    return len(trace["entropy_by_layer"])


def fixed_exit(trace: dict, fraction: float) -> int:
    return min(len(trace["top_id_by_layer"]), max(1, math.ceil(len(trace["top_id_by_layer"]) * fraction)))


def kstable_exit(trace: dict, base_exit: int, min_fraction: float, k: int) -> int:
    if trace["criticality"] < 0.8:
        return base_exit
    min_layer = fixed_exit(trace, min_fraction)
    start = max(base_exit, min_layer, k)
    top_ids = trace["top_id_by_layer"]
    for layer in range(start, len(top_ids) + 1):
        if len(set(top_ids[layer - k : layer])) == 1:
            return layer
    return len(top_ids)


def disagrees(trace: dict, exit_layer: int) -> bool:
    return not trace["pred_matches_final_by_layer"][exit_layer - 1]


def text(x: float, y: float, value: object, size: int = 14, weight: int = 500, fill: str = "ink", anchor: str = "start") -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="IBM Plex Mono, ui-monospace, monospace" '
        f'font-size="{size}" font-weight="{weight}" fill="{COLORS[fill]}" text-anchor="{anchor}">{esc(value)}</text>'
    )


def serif(x: float, y: float, value: object, size: int = 22, weight: int = 600, fill: str = "ink") -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="JetBrains Mono, IBM Plex Mono, ui-monospace, monospace" letter-spacing="-0.5" '
        f'font-size="{size}" font-weight="{weight}" fill="{COLORS[fill]}">{esc(value)}</text>'
    )


def svg(width: int, height: int, body: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">
  <rect width="{width}" height="{height}" fill="{COLORS['paper']}"/>
  {body}
</svg>
"""


def save(name: str, content: str) -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    (ASSET_DIR / name).write_text(content, encoding="utf-8")


def make_pareto(rows: list[dict[str, str]]) -> None:
    selected = [
        ("entropy", "threshold=0.10", "Entropy", "red"),
        ("margin", "threshold=0.50", "Margin", "blue"),
        ("tokentriage-kstable", "entropy=0.10,min=0.85,k=4", "TokenTriage", "teal"),
        ("tokentriage-kstable", "entropy=0.20,min=0.85,k=4", "Triage high-save", "green"),
    ]
    found = []
    for policy, setting, label, color in selected:
        row = next(r for r in rows if r["policy"] == policy and r["setting"] == setting)
        found.append((row, label, color))

    def px(saved: float) -> float:
        return 110 + saved / 0.9 * 720

    def py(disagree: float) -> float:
        return 410 - disagree / 0.95 * 320

    parts = [
        serif(48, 48, "Compute-safety frontier", 28),
        text(50, 78, "lower is safer · farther right is cheaper", 13, 500, "muted"),
        '<rect x="100" y="90" width="750" height="340" rx="8" fill="#fbfbf8" stroke="#d8d8d0"/>',
    ]
    for i in range(5):
        y = 90 + i * 68
        x = 100 + i * 150
        parts.append(f'<path d="M100 {y}H850" stroke="{COLORS["line"]}" stroke-width="1"/>')
        parts.append(f'<path d="M{x} 90V430" stroke="{COLORS["line"]}" stroke-width="1"/>')
    parts.append(f'<path d="M130 384 C260 310 392 270 538 235 C660 205 756 150 828 104" fill="none" stroke="#d9e4df" stroke-width="8" stroke-linecap="round"/>')
    for row, label, color in found:
        x = px(float(row["compute_saved"]))
        y = py(float(row["disagreement_critical"]))
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="13" fill="{COLORS[color]}" stroke="#fff" stroke-width="4"/>')
        parts.append(text(x + 18, y - 4, label, 13, 700))
        parts.append(text(x + 18, y + 15, f'{float(row["disagreement_critical"]) * 100:.1f}% crit.', 12, 500, "muted"))
    parts += [
        text(450, 470, "simulated layer-compute saved", 14, 700, "muted", "middle"),
        f'<text x="-292" y="36" transform="rotate(-90)" font-family="IBM Plex Mono, monospace" font-size="14" font-weight="700" fill="{COLORS["muted"]}" text-anchor="middle">critical-token disagreement</text>',
        text(100, 454, "0", 12, 600, "muted", "middle"),
        text(850, 454, "0.9", 12, 600, "muted", "middle"),
        text(78, 430, "0", 12, 600, "muted", "end"),
        text(78, 96, "0.95", 12, 600, "muted", "end"),
    ]
    save("pareto.svg", svg(900, 500, "\n  ".join(parts)))


def make_mix(traces: list[dict]) -> None:
    counts = Counter(t["category"] for t in traces if t["criticality"] >= 0.8)
    order = ["negation", "temporality", "laterality", "finding", "severity", "measurement", "measurement_unit"]
    max_count = max(counts.values())
    parts = [serif(42, 48, "Critical token mix", 28), text(44, 76, "341 tagged tokens in 64 reports", 13, 500, "muted")]
    y = 112
    for cat in order:
        if counts[cat] == 0:
            continue
        w = counts[cat] / max_count * 560
        parts.append(text(48, y + 18, cat.replace("_", " "), 13, 700))
        parts.append(f'<rect x="210" y="{y}" width="560" height="24" rx="4" fill="#f2f2ec" stroke="{COLORS["line"]}"/>')
        parts.append(f'<rect x="210" y="{y}" width="{w:.1f}" height="24" rx="4" fill="{COLORS["teal"]}"/>')
        parts.append(text(790, y + 18, counts[cat], 13, 700, "ink"))
        y += 42
    save("critical_mix.svg", svg(860, 430, "\n  ".join(parts)))


def make_gap(rows: list[dict[str, str]]) -> None:
    entropy = next(r for r in rows if r["policy"] == "entropy" and r["setting"] == "threshold=0.10")
    triage = next(r for r in rows if r["policy"] == "tokentriage-kstable" and r["setting"] == "entropy=0.10,min=0.85,k=4")
    e = float(entropy["disagreement_critical"])
    t = float(triage["disagreement_critical"])
    parts = [
        serif(42, 46, "The gap TokenTriage closes", 28),
        text(44, 76, "critical-token disagreement at comparable compute", 13, 500, "muted"),
        f'<rect x="80" y="132" width="740" height="46" rx="6" fill="#f2f2ec" stroke="{COLORS["line"]}"/>',
        f'<rect x="80" y="132" width="{740 * e:.1f}" height="46" rx="6" fill="{COLORS["red"]}"/>',
        text(92, 163, "entropy baseline", 14, 700, "paper"),
        text(80 + 740 * e + 16, 162, f"{e*100:.1f}%", 22, 700, "red"),
        f'<rect x="80" y="236" width="740" height="46" rx="6" fill="#f2f2ec" stroke="{COLORS["line"]}"/>',
        f'<rect x="80" y="236" width="{740 * t:.1f}" height="46" rx="6" fill="{COLORS["teal"]}"/>',
        text(92, 267, "TokenTriage-kstable", 14, 700, "paper"),
        text(80 + 740 * t + 16, 266, f"{t*100:.1f}%", 22, 700, "teal"),
        f'<path d="M{80 + 740*t:.1f} 314H{80 + 740*e:.1f}" stroke="{COLORS["ink"]}" stroke-width="2" marker-end="url(#arrowGap)"/>',
        text((80 + 740*t + 80 + 740*e) / 2, 344, f"{(e-t)*100:.1f} point reduction", 14, 700, "ink", "middle"),
    ]
    body = """
  <defs><marker id="arrowGap" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#172023"/></marker></defs>
  """ + "\n  ".join(parts)
    save("gap.svg", svg(900, 390, body))


def make_examples(traces: list[dict]) -> None:
    rows_by_category: dict[str, list[tuple[dict, int, int]]] = {}
    for t in traces:
        if t["criticality"] < 0.8:
            continue
        e_exit = entropy_exit(t, 0.10)
        k_exit = kstable_exit(t, e_exit, 0.85, 4)
        if disagrees(t, e_exit) and not disagrees(t, k_exit):
            rows_by_category.setdefault(t["category"], []).append((t, e_exit, k_exit))
    priority = {"negation": 0, "laterality": 1, "finding": 2, "measurement": 3, "severity": 4, "temporality": 5}
    rows: list[tuple[dict, int, int]] = []
    for category in sorted(rows_by_category, key=lambda cat: priority.get(cat, 9)):
        choices = sorted(rows_by_category[category], key=lambda item: (item[0]["report_id"], item[0]["position"]))
        rows.append(choices[0])
        if len(rows) >= 6:
            break
    if len(rows) < 6:
        extras = [
            item
            for category_rows in rows_by_category.values()
            for item in category_rows
            if item not in rows
        ]
        rows.extend(sorted(extras, key=lambda item: (priority.get(item[0]["category"], 9), item[0]["report_id"], item[0]["position"]))[: 6 - len(rows)])
    rows.sort(key=lambda item: (priority.get(item[0]["category"], 9), item[0]["report_id"], item[0]["position"]))
    parts = [serif(42, 46, "Real rescued tokens", 28), text(44, 76, "entropy exits disagree; TokenTriage waits and matches the final model", 13, 500, "muted")]
    y = 116
    for t, e_exit, k_exit in rows:
        token = t["token_text"].strip() or t["token_text"]
        parts.append(f'<rect x="44" y="{y-24}" width="812" height="46" rx="6" fill="#fbfbf8" stroke="{COLORS["line"]}"/>')
        parts.append(text(62, y + 5, f'{token!r}', 15, 700, "red"))
        parts.append(text(190, y + 5, t["category"], 13, 700, "teal"))
        parts.append(text(340, y + 5, f'entropy L{e_exit}', 13, 600, "muted"))
        parts.append(text(500, y + 5, f'TokenTriage L{k_exit}', 13, 600, "ink"))
        parts.append(text(708, y + 5, t["report_id"].replace(".png", ""), 12, 500, "muted"))
        y += 58
    save("examples.svg", svg(900, 500, "\n  ".join(parts)))
    (ASSET_DIR / "examples.json").write_text(
        json.dumps(
            [
                {
                    "token": item[0]["token_text"].strip(),
                    "category": item[0]["category"],
                    "report_id": item[0]["report_id"],
                    "entropy_exit": item[1],
                    "triage_exit": item[2],
                }
                for item in rows
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def best_entropy(rows: list[dict[str, str]]) -> dict[str, str]:
    return min((row for row in rows if row["policy"] == "entropy"), key=lambda row: float(row["disagreement_critical"]))


def best_margin(rows: list[dict[str, str]]) -> dict[str, str]:
    return min((row for row in rows if row["policy"] == "margin"), key=lambda row: float(row["disagreement_critical"]))


def best_deployable_triage(rows: list[dict[str, str]], preferred_setting: str | None = None) -> dict[str, str]:
    candidates = [row for row in rows if row["policy"] in {"tokentriage-minlayer", "tokentriage-kstable"}]
    if preferred_setting:
        preferred = [row for row in candidates if row["setting"] == preferred_setting]
        if preferred:
            return preferred[0]
    return min(candidates, key=lambda row: (float(row["disagreement_critical"]), -float(row["compute_saved"])))


def make_validation() -> None:
    available = [(label, path, preferred) for label, path, preferred in EXTENDED_RUNS if (path / "policy_results.csv").exists()]
    if not available:
        return
    parts = [
        serif(42, 46, "Cross-run validation", 28),
        text(44, 76, "best confidence baseline vs deployable TokenTriage", 13, 500, "muted"),
    ]
    summary: list[dict[str, object]] = []
    y = 124
    for label, path, preferred in available:
        rows = read_policy_rows_from(path)
        entropy = best_entropy(rows)
        margin = best_margin(rows)
        triage = best_deployable_triage(rows, preferred)
        baseline = min([entropy, margin], key=lambda row: float(row["disagreement_critical"]))
        meta = read_meta(path)
        base_width = float(baseline["disagreement_critical"]) * 460
        triage_width = float(triage["disagreement_critical"]) * 460
        parts.append(text(54, y - 14, label, 14, 700, "ink"))
        parts.append(f'<rect x="300" y="{y-30}" width="460" height="20" rx="4" fill="#f2f2ec" stroke="{COLORS["line"]}"/>')
        parts.append(f'<rect x="300" y="{y-30}" width="{base_width:.1f}" height="20" rx="4" fill="{COLORS["red"]}"/>')
        parts.append(f'<rect x="300" y="{y+2}" width="460" height="20" rx="4" fill="#f2f2ec" stroke="{COLORS["line"]}"/>')
        parts.append(f'<rect x="300" y="{y+2}" width="{triage_width:.1f}" height="20" rx="4" fill="{COLORS["teal"]}"/>')
        parts.append(text(772, y - 14, f'{float(baseline["disagreement_critical"]) * 100:.1f}% baseline', 12, 700, "red"))
        parts.append(text(772, y + 18, f'{float(triage["disagreement_critical"]) * 100:.1f}% triage', 12, 700, "teal"))
        parts.append(text(54, y + 20, f'{meta.get("limit", "?")} reports · {triage["compute_saved"]} compute saved', 12, 500, "muted"))
        summary.append(
            {
                "label": label,
                "dataset": meta.get("dataset"),
                "split": meta.get("split"),
                "model": meta.get("model"),
                "reports": meta.get("limit"),
                "tokens": int(triage["tokens"]),
                "critical_tokens": int(triage["critical_tokens"]),
                "baseline_policy": baseline["policy"],
                "baseline_setting": baseline["setting"],
                "baseline_critical_disagreement": float(baseline["disagreement_critical"]),
                "triage_policy": triage["policy"],
                "triage_setting": triage["setting"],
                "triage_compute_saved": float(triage["compute_saved"]),
                "triage_critical_disagreement": float(triage["disagreement_critical"]),
            }
        )
        y += 92
    save("validation.svg", svg(900, max(250, y + 28), "\n  ".join(parts)))
    (ASSET_DIR / "validation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def make_architecture() -> None:
    parts = [
        serif(42, 46, "Evaluation pipeline", 28),
        text(44, 76, "teacher forcing  ·  logit lens  ·  risk tags  ·  routing policy  ·  harm-weighted metrics", 13, 500, "muted"),
    ]
    labels = [
        ("reports", "64 IU X-Ray reports"),
        ("traces", "3,783 layer traces"),
        ("risk", "341 critical tokens"),
        ("policies", "137 policy settings"),
        ("metrics", "bootstrap CIs"),
    ]
    x = 56
    for i, (head, sub) in enumerate(labels):
        parts.append(f'<rect x="{x}" y="134" width="142" height="96" rx="8" fill="#fbfbf8" stroke="{COLORS["line"]}"/>')
        parts.append(text(x + 71, 174, head, 15, 700, "ink", "middle"))
        parts.append(text(x + 71, 199, sub, 11, 500, "muted", "middle"))
        if i < len(labels) - 1:
            parts.append(f'<path d="M{x+142} 182H{x+188}" stroke="{COLORS["ink"]}" stroke-width="1.8" marker-end="url(#arrowArch)"/>')
        x += 188
    body = """
  <defs><marker id="arrowArch" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#172023"/></marker></defs>
  """ + "\n  ".join(parts)
    save("architecture.svg", svg(970, 290, body))


def main() -> None:
    rows = read_policy_rows()
    traces = read_traces()
    make_pareto(rows)
    make_mix(traces)
    make_gap(rows)
    make_examples(traces)
    make_validation()
    make_architecture()
    print(f"Wrote SVG figures to {ASSET_DIR}")


if __name__ == "__main__":
    main()
