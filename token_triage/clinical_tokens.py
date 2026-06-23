from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenSignal:
    token: str
    index: int
    category: str
    criticality: float


TOKEN_RE = re.compile(r"\d+(?:\.\d+)?|[A-Za-z]+(?:[-'][A-Za-z]+)?|[^\w\s]", re.UNICODE)

CRITICAL_PATTERNS: list[tuple[str, re.Pattern[str], float]] = [
    ("negation", re.compile(r"^(no|not|without|negative|denies|absent|free)$", re.I), 0.98),
    ("laterality", re.compile(r"^(left|right|bilateral|unilateral|upper|lower|apical|basilar)$", re.I), 0.88),
    ("severity", re.compile(r"^(mild|moderate|severe|marked|trace|small|large|worsening|improved)$", re.I), 0.82),
    ("temporality", re.compile(r"^(acute|chronic|new|old|prior|interval|stable|persistent|resolved)$", re.I), 0.84),
    ("medication", re.compile(r"^(heparin|warfarin|insulin|tpa|alteplase|aspirin|anticoagulation)$", re.I), 0.92),
    ("measurement", re.compile(r"^\d+(?:\.\d+)?$", re.I), 0.9),
    ("finding", re.compile(r"^(pneumothorax|effusion|opacity|edema|consolidation|hemorrhage|infarct|fracture|mass|nodule)$", re.I), 0.9),
]


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def detect_critical_tokens(text: str) -> list[TokenSignal]:
    signals: list[TokenSignal] = []
    for index, token in enumerate(tokenize(text)):
        clean = token.strip(".,;:()[]{}").lower()
        for category, pattern, criticality in CRITICAL_PATTERNS:
            if pattern.match(clean):
                signals.append(TokenSignal(token=token, index=index, category=category, criticality=criticality))
                break
        if re.match(r"^(cm|mm|mg|ml|units?)$", clean):
            signals.append(TokenSignal(token=token, index=index, category="measurement_unit", criticality=0.88))
    return signals


def token_is_critical(token: str) -> tuple[str, float] | None:
    clean = token.strip(".,;:()[]{}").lower()
    for category, pattern, criticality in CRITICAL_PATTERNS:
        if pattern.match(clean):
            return category, criticality
    if re.match(r"^(cm|mm|mg|ml|units?)$", clean):
        return "measurement_unit", 0.88
    return None


# --- pluggable risk taggers -------------------------------------------------
# A risk tagger maps a token string to (category, criticality) or None. The
# default is the clinical lexicon above; swap in RadGraph/CheXbert/radiologist
# annotations by registering another tagger with the same signature.
from typing import Callable, Optional  # noqa: E402

RiskTagger = Callable[[str], Optional[tuple]]

_TAGGERS: dict[str, RiskTagger] = {"lexicon": token_is_critical}


def register_tagger(name: str, tagger: RiskTagger) -> None:
    """Register a custom risk tagger (e.g. a RadGraph- or CheXbert-backed one)."""
    _TAGGERS[name] = tagger


def get_tagger(name: str = "lexicon") -> RiskTagger:
    if name not in _TAGGERS:
        raise KeyError(f"Unknown risk tagger '{name}'. Registered: {sorted(_TAGGERS)}")
    return _TAGGERS[name]


def available_taggers() -> list[str]:
    return sorted(_TAGGERS)
