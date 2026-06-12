"""Ollama client with schema-enforced JSON outputs (PLAN.md S4).

Every LLM call passes a pydantic model's JSON schema as Ollama's `format`
parameter (grammar-constrained decoding), then validates the response with
that same model. Validation failures retry; exhausted retries or an
unreachable server raise LLMError — callers (agents) convert that into the
standard neutral zero-confidence fallback, never a guessed score.
"""

from __future__ import annotations

import logging
from typing import TypeVar

import requests
from pydantic import BaseModel, ValidationError

from trading_platform.core.config import LLMSettings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """LLM unavailable or unable to produce schema-valid output."""


class OpenAICompatClient:
    """Chat-completions client for any OpenAI-compatible endpoint (OpenAI,
    Anthropic's compat API, Groq, a remote Ollama's /v1, ...).

    External APIs don't all support grammar-constrained decoding, so the
    schema goes into the prompt and the response is pydantic-validated with
    retries — same contract as OllamaClient.generate()."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout_seconds: int = 120):
        if not base_url or not model:
            raise LLMError("OpenAI-compatible client needs base_url and model")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout_seconds

    def generate(
        self,
        prompt: str,
        response_model: type[T],
        system: str | None = None,
        retries: int = 2,
    ) -> T:
        schema = response_model.model_json_schema()
        sys_text = (system or "") + (
            "\n\nRespond with ONLY a valid JSON object matching this JSON schema "
            f"(no prose, no code fences):\n{schema}"
        )
        messages = [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": prompt},
        ]

        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model, "messages": messages,
                          "temperature": 0.2},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
            except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
                raise LLMError(f"endpoint unreachable at {self.base_url}: {exc}") from exc

            try:
                return response_model.model_validate_json(_strip_fences(content))
            except ValidationError as exc:
                last_error = exc
                logger.warning(
                    "external LLM output failed %s validation (attempt %d/%d)",
                    response_model.__name__, attempt + 1, retries + 1,
                )

        raise LLMError(
            f"external LLM could not produce valid {response_model.__name__} "
            f"after {retries + 1} attempts: {last_error}"
        )


def _strip_fences(content: str) -> str:
    """Models sometimes wrap JSON in markdown fences despite instructions."""
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[: -3]
        text = text.removeprefix("json").strip()
    return text


class OllamaClient:
    def __init__(self, settings: LLMSettings):
        self.settings = settings

    def generate(
        self,
        prompt: str,
        response_model: type[T],
        system: str | None = None,
        retries: int = 2,
    ) -> T:
        """Generate a schema-valid response, retrying on validation failure."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.settings.model,
            "messages": messages,
            "stream": False,
            "format": response_model.model_json_schema(),
            "options": {"temperature": 0.1},
        }

        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                resp = requests.post(
                    f"{self.settings.base_url}/api/chat",
                    json=payload,
                    timeout=self.settings.timeout_seconds,
                )
                resp.raise_for_status()
                content = resp.json()["message"]["content"]
            except (requests.RequestException, KeyError, ValueError) as exc:
                # Server down/misbehaving: retrying won't help within this run.
                raise LLMError(f"ollama unreachable at {self.settings.base_url}: {exc}") from exc

            try:
                return response_model.model_validate_json(content)
            except ValidationError as exc:
                last_error = exc
                logger.warning(
                    "LLM output failed %s validation (attempt %d/%d)",
                    response_model.__name__, attempt + 1, retries + 1,
                )

        raise LLMError(
            f"LLM could not produce valid {response_model.__name__} "
            f"after {retries + 1} attempts: {last_error}"
        )
