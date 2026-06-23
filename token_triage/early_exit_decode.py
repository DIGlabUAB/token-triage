"""Real autoregressive early-exit decoding (free-running, with error propagation).

Unlike the teacher-forced trace analysis, this actually GENERATES text: each emitted
token is chosen at an early layer and fed back, so an early mistake propagates through
the rest of the generation exactly as it would at deployment.

The model is run to full depth internally (so we need no manual KV/layer surgery), but
the token we EMIT is taken from the chosen exit layer's logit-lens projection, and we
record the layer that *would* have been executed. This yields a faithful early-exit
generation plus an honest compute-saved measurement, while staying architecture-agnostic.

TokenTriage uses speculative risk gating: take the cheap confidence-exit token; if that
token is clinically critical, keep going until the prediction is stable for k layers
(and past a minimum depth) before committing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from .clinical_tokens import token_is_critical


@dataclass
class Generation:
    mode: str
    text: str
    tokens: list[int]
    exit_layers: list[int]
    n_layers: int
    overrides: list[dict] | None = None  # critical-token interventions (triage mode)

    @property
    def compute_saved(self) -> float:
        if not self.exit_layers:
            return 0.0
        return 1.0 - (sum(self.exit_layers) / len(self.exit_layers)) / self.n_layers


def _project(model, hidden_vec, is_last_layer):
    if not is_last_layer and hasattr(model, "model") and hasattr(model.model, "norm"):
        hidden_vec = model.model.norm(hidden_vec)
    return model.lm_head(hidden_vec).float()


def generate_early_exit(
    model,
    tokenizer,
    torch,
    prompt_ids,
    *,
    mode: str = "triage",
    entropy_thr: float = 0.10,
    min_frac: float = 0.85,
    k: int = 4,
    max_new: int = 80,
    risk_tagger: Callable[[str], object] | None = None,
) -> Generation:
    """mode in {'full', 'entropy', 'triage'}."""
    risk_tagger = risk_tagger or token_is_critical
    device = prompt_ids.device
    seq = prompt_ids
    exit_layers: list[int] = []
    emitted: list[int] = []
    overrides: list[dict] = []
    vocab_norm = math.log(float(model.config.vocab_size))
    eos = tokenizer.eos_token_id
    n_layers = int(model.config.num_hidden_layers)

    with torch.inference_mode():
        for _ in range(max_new):
            out = model(input_ids=seq, output_hidden_states=True)
            hs = out.hidden_states[1:]
            n = len(hs)
            if mode == "full":
                tok = int(out.logits[0, -1, :].float().argmax())
                exit_layers.append(n)
            else:
                tops: list[int] = []
                base = n
                for li, h in enumerate(hs):
                    logits = _project(model, h[0, -1, :], li == n - 1)
                    probs = torch.softmax(logits, dim=-1)
                    log_probs = torch.log_softmax(logits, dim=-1)
                    ent = float((-(probs * log_probs).sum()) / vocab_norm)
                    tops.append(int(logits.argmax()))
                    if base == n and ent <= entropy_thr:
                        base = li + 1
                        if mode == "entropy":
                            break
                exit_layer = base
                tok = tops[base - 1]
                if mode == "triage":
                    cheap_tok = tok
                    cand = tokenizer.decode([cheap_tok]).strip()
                    if risk_tagger(cand):
                        start = max(base, max(1, math.ceil(n * min_frac)), k)
                        exit_layer = n
                        for layer in range(start, n + 1):
                            if len(set(tops[layer - k:layer])) == 1:
                                exit_layer = layer
                                break
                        tok = tops[exit_layer - 1]
                        overrides.append({
                            "step": len(emitted),
                            "cheap_token": cand,
                            "committed_token": tokenizer.decode([tok]).strip(),
                            "cheap_layer": base,
                            "committed_layer": exit_layer,
                            "changed": tok != cheap_tok,
                        })
                exit_layers.append(exit_layer)
            emitted.append(tok)
            seq = torch.cat([seq, torch.tensor([[tok]], device=device)], dim=1)
            if eos is not None and tok == eos:
                break

    text = tokenizer.decode(emitted, skip_special_tokens=True)
    return Generation(mode=mode, text=text, tokens=emitted, exit_layers=exit_layers,
                      n_layers=n_layers, overrides=overrides if mode == "triage" else None)
