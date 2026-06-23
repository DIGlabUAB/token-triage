"""Real early-exit DECODING experiment (free-running, error propagates).

Generates continuations of public radiology reports under three decoders -- full depth,
confidence-only entropy exit, and TokenTriage -- then measures how faithfully each
early-exit generation preserves the full model's critical clinical content, alongside the
actual compute saved during generation.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from token_triage.clinical_tokens import detect_critical_tokens
from token_triage.early_exit_decode import generate_early_exit
from token_triage.hf_dataset import fetch_hf_reports


def crit_multiset(text: str) -> Counter:
    return Counter((s.category, s.token.strip(".,;:()[]{}").lower()) for s in detect_critical_tokens(text))


def fidelity(ref: Counter, hyp: Counter) -> float:
    if not ref:
        return 1.0
    overlap = sum((ref & hyp).values())
    return overlap / sum(ref.values())


def prefix_agreement(ref_tokens: list[int], hyp_tokens: list[int]) -> float:
    m = min(len(ref_tokens), len(hyp_tokens))
    if m == 0:
        return 1.0
    same = sum(1 for a, b in zip(ref_tokens[:m], hyp_tokens[:m]) if a == b)
    return same / m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dataset", default="ChayanM/IUXray-Data-Train-Test")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--prompt-tokens", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=60)
    ap.add_argument("--entropy-thr", type=float, default=0.10)
    ap.add_argument("--output-dir", default="outputs/real-decoding-iuxray-qwen15b")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16 if device == "mps" else torch.float32,
        trust_remote_code=True, low_cpu_mem_usage=True,
    ).to(device).eval()

    reports = fetch_hf_reports(dataset=args.dataset, split=args.split, limit=args.limit)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def build_prompt(seed_text: str):
        msgs = [{"role": "user", "content":
                 "You are a radiologist. Continue this chest X-ray report in one or two "
                 "sentences, using standard radiology language:\n" + seed_text}]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = seed_text
        return tok(text, return_tensors="pt").input_ids.to(device)

    records = []
    agg = {m: {"fid": [], "saved": [], "prefix": []} for m in ("entropy", "triage")}
    override_total = 0
    override_changed = 0
    examples = []
    override_examples = []
    for i, rep in enumerate(reports):
        seed = " ".join(rep.text.split()[: args.prompt_tokens])
        prompt = build_prompt(seed)
        gens = {}
        for mode in ("full", "entropy", "triage"):
            gens[mode] = generate_early_exit(
                model, tok, torch, prompt, mode=mode,
                entropy_thr=args.entropy_thr, max_new=args.max_new,
            )
        ref_ms = crit_multiset(gens["full"].text)
        # harm-gating interventions during triage decoding
        ovs = gens["triage"].overrides or []
        override_total += len(ovs)
        override_changed += sum(1 for o in ovs if o["changed"])
        for o in ovs:
            if o["changed"]:
                override_examples.append({"report_id": rep.report_id, **o})
        row = {
            "report_id": rep.report_id,
            "seed": seed,
            "full_text": gens["full"].text,
            "triage_critical_overrides": len(ovs),
            "triage_overrides_changed": sum(1 for o in ovs if o["changed"]),
        }
        for mode in ("entropy", "triage"):
            fid = fidelity(ref_ms, crit_multiset(gens[mode].text))
            saved = gens[mode].compute_saved
            pref = prefix_agreement(gens["full"].tokens, gens[mode].tokens)
            agg[mode]["fid"].append(fid)
            agg[mode]["saved"].append(saved)
            agg[mode]["prefix"].append(pref)
            row[f"{mode}_text"] = gens[mode].text
            row[f"{mode}_crit_fidelity"] = round(fid, 4)
            row[f"{mode}_compute_saved"] = round(saved, 4)
            row[f"{mode}_prefix_agreement"] = round(pref, 4)
        records.append(row)
        # qualitative: entropy lost critical content that triage kept
        if row["entropy_crit_fidelity"] < row["triage_crit_fidelity"]:
            examples.append({k: row[k] for k in ("report_id", "full_text", "entropy_text", "triage_text",
                                                 "entropy_crit_fidelity", "triage_crit_fidelity")})
        print(f"[{i+1}/{len(reports)}] {rep.report_id[:24]:24} "
              f"ent fid={row['entropy_crit_fidelity']:.2f} save={row['entropy_compute_saved']:.2f} | "
              f"tri fid={row['triage_crit_fidelity']:.2f} save={row['triage_compute_saved']:.2f}", flush=True)

    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    summary = {
        "model": args.model, "dataset": args.dataset, "reports": len(records),
        "prompt_tokens": args.prompt_tokens, "max_new": args.max_new, "entropy_thr": args.entropy_thr,
        "entropy_crit_fidelity": mean(agg["entropy"]["fid"]),
        "triage_crit_fidelity": mean(agg["triage"]["fid"]),
        "entropy_compute_saved": mean(agg["entropy"]["saved"]),
        "triage_compute_saved": mean(agg["triage"]["saved"]),
        "entropy_prefix_agreement": mean(agg["entropy"]["prefix"]),
        "triage_prefix_agreement": mean(agg["triage"]["prefix"]),
        "triage_critical_overrides": override_total,
        "triage_overrides_changed_token": override_changed,
        "triage_override_change_rate": round(override_changed / override_total, 4) if override_total else 0.0,
    }
    (out_dir / "decoding_records.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    (out_dir / "decoding_examples.json").write_text(json.dumps(examples[:8], indent=2) + "\n", encoding="utf-8")
    (out_dir / "decoding_overrides.json").write_text(json.dumps(override_examples[:20], indent=2) + "\n", encoding="utf-8")
    (out_dir / "decoding_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("\n=== REAL DECODING SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
