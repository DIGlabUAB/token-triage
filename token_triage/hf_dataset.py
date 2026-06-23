from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass


HF_DATASET_VIEWER = "https://datasets-server.huggingface.co"


def _get_json(url: str, *, retries: int = 6, timeout: int = 60) -> dict:
    """GET a JSON payload with exponential backoff on rate limits / transient errors."""
    delay = 3.0
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "token-triage/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                wait = delay
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                if retry_after and retry_after.isdigit():
                    wait = max(wait, float(retry_after))
                time.sleep(wait)
                delay = min(delay * 2, 60.0)
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
                continue
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError("unreachable")


@dataclass(frozen=True)
class HfReport:
    row_id: int
    report_id: str
    text: str
    dataset: str
    split: str


def fetch_hf_reports(
    *,
    dataset: str = "ChayanM/IUXray-Data-Train-Test",
    config: str = "default",
    split: str = "test",
    limit: int = 64,
    offset: int = 0,
    text_column: str = "Caption",
    id_column: str = "Image_Name",
) -> list[HfReport]:
    reports: list[HfReport] = []
    cursor = offset  # advances by rows fetched, independent of how many pass the text filter
    seen_columns: set[str] = set()
    while len(reports) < limit:
        page_len = min(100, limit - len(reports))
        params = urllib.parse.urlencode(
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": cursor,
                "length": page_len,
            }
        )
        payload = _get_json(f"{HF_DATASET_VIEWER}/rows?{params}")
        rows = payload.get("rows", [])
        if not rows:
            break
        cursor += len(rows)
        for item in rows:
            row = item["row"]
            seen_columns.update(row.keys())
            text = str(row.get(text_column, "")).strip()
            if not text:
                continue
            reports.append(
                HfReport(
                    row_id=int(item["row_idx"]),
                    report_id=str(row.get(id_column, item["row_idx"])),
                    text=text,
                    dataset=dataset,
                    split=split,
                )
            )
            if len(reports) >= limit:
                break
    if not reports and seen_columns:
        raise ValueError(
            f"No rows had non-empty text in column '{text_column}'. "
            f"Available columns: {sorted(seen_columns)}. Pass --text-column accordingly."
        )
    return reports
