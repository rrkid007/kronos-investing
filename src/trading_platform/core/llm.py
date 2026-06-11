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
