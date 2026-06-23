from __future__ import annotations

from dataclasses import dataclass

from .ollama import OllamaClient


@dataclass(frozen=True)
class Case:
    case_id: str
    source: str
    report: str


SEED_REPORTS = [
    "No acute cardiopulmonary abnormality. Mild left basilar opacity likely reflects atelectasis. No pneumothorax.",
    "There is a new 1.8 cm right upper lobe nodule. Small left pleural effusion is present without edema.",
    "Severe bilateral airspace consolidation has worsened since the prior exam. No evidence of pneumothorax.",
    "Chronic right basal scarring is stable. The heart size is mildly enlarged. No focal opacity.",
    "Moderate left hydronephrosis with a 4 mm distal ureteral stone. No right-sided obstruction.",
    "Acute infarct is not identified. Old left frontal encephalomalacia is unchanged. No hemorrhage.",
    "Trace right apical pneumothorax measures 6 mm. Left chest tube remains in place.",
    "Large right pleural effusion with interval worsening. No mediastinal shift.",
]


def template_cases(limit: int) -> list[Case]:
    cases: list[Case] = []
    while len(cases) < limit:
        idx = len(cases) % len(SEED_REPORTS)
        cases.append(Case(case_id=f"template-{len(cases) + 1:03d}", source="template", report=SEED_REPORTS[idx]))
    return cases


def ollama_cases(client: OllamaClient, limit: int) -> list[Case]:
    prompt = f"""
Generate {limit} concise synthetic radiology impression lines.
Requirements:
- One line per case.
- Include clinically meaningful tokens such as negation, laterality, severity, temporality, medications, or measurements.
- Do not include patient identifiers.
- Return only the numbered lines.
"""
    raw = client.generate(prompt, temperature=0.35, num_predict=600)
    reports: list[str] = []
    for line in raw.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        cleaned = cleaned.lstrip("-* ")
        if ". " in cleaned[:5]:
            cleaned = cleaned.split(". ", 1)[1].strip()
        if len(cleaned.split()) >= 5:
            reports.append(cleaned)
    if len(reports) < limit:
        reports.extend(case.report for case in template_cases(limit - len(reports)))
    return [Case(case_id=f"ollama-{i + 1:03d}", source="ollama", report=report) for i, report in enumerate(reports[:limit])]
