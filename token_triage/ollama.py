from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OllamaConfig:
    host: str = "http://127.0.0.1:11434"
    model: str = "qwen2.5:1.5b"
    timeout_seconds: int = 120


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, config: OllamaConfig):
        self.config = config

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.config.host}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise OllamaError(f"Could not reach Ollama at {self.config.host}: {exc}") from exc

    def _get_json(self, path: str) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(f"{self.config.host}{path}", timeout=20) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise OllamaError(f"Could not reach Ollama at {self.config.host}: {exc}") from exc

    def available_models(self) -> list[str]:
        payload = self._get_json("/api/tags")
        return [model["name"] for model in payload.get("models", [])]

    def generate(self, prompt: str, *, temperature: float = 0.2, num_predict: int = 256) -> str:
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": num_predict},
        }
        data = self._post_json("/api/generate", payload)
        return str(data.get("response", "")).strip()

    def supports_logprobs(self) -> bool:
        payload = {
            "model": self.config.model,
            "prompt": "No acute cardiopulmonary abnormality.",
            "max_tokens": 3,
            "logprobs": 3,
            "temperature": 0,
        }
        try:
            data = self._post_json("/v1/completions", payload)
        except OllamaError:
            return False
        choices = data.get("choices", [])
        return bool(choices and choices[0].get("logprobs"))
