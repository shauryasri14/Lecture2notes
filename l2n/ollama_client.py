"""Minimal Ollama HTTP client with explicit timeouts and bounded retries."""
from __future__ import annotations

import logging
import time
from typing import Callable

import requests

from .errors import (
    LLMOutputError,
    OllamaError,
    OllamaModelMissingError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)

log = logging.getLogger("l2n.ollama")


class OllamaClient:
    def __init__(
        self,
        host: str,
        model: str,
        *,
        timeout: float = 600.0,
        max_retries: int = 3,
        temperature: float = 0.2,
        num_ctx: int = 8192,
        num_predict: int = 4096,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        backoff: float = 2.0,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.session = session or requests.Session()
        self._sleep = sleep
        self.backoff = backoff

    # -- availability -----------------------------------------------------
    def check_available(self) -> None:
        """Fail fast (no retries) if the server or model is unavailable."""
        try:
            resp = self.session.get(f"{self.host}/api/tags", timeout=10)
        except (requests.ConnectionError, requests.Timeout) as exc:
            raise OllamaUnavailableError(
                f"Cannot reach Ollama at {self.host}. Is it running? "
                "Start it with `ollama serve` (or launch the Ollama app)."
            ) from exc
        if resp.status_code != 200:
            raise OllamaUnavailableError(f"Ollama at {self.host} answered HTTP {resp.status_code} to /api/tags.")
        try:
            models = resp.json().get("models", [])
        except ValueError as exc:
            raise OllamaUnavailableError(f"{self.host} did not return valid JSON; is it really Ollama?") from exc
        names = {m.get("name") or m.get("model") for m in models}
        wanted = {self.model, self.model + ":latest"} if ":" not in self.model else {self.model}
        if not names & wanted:
            raise OllamaModelMissingError(
                f"Model '{self.model}' is not installed in Ollama. Run: ollama pull {self.model}"
            )

    # -- generation -------------------------------------------------------
    def generate(self, prompt: str, system: str | None = None, json_mode: bool = False) -> str:
        """Return the model's text. Retries transient failures at most
        `max_retries` times (so at most max_retries+1 requests)."""
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            },
        }
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"

        last: OllamaError | None = None
        for attempt in range(self.max_retries + 1):
            if attempt:
                delay = self.backoff * (2 ** (attempt - 1))
                log.warning("Retrying Ollama request (%d/%d) in %.0fs: %s", attempt, self.max_retries, delay, last)
                self._sleep(delay)
            try:
                return self._post(payload)
            except _NonRetryable as exc:
                raise exc.inner from None
            except (OllamaTimeoutError, OllamaUnavailableError, LLMOutputError) as exc:
                last = exc  # transient: retry
        assert last is not None
        raise last

    def _post(self, payload: dict) -> str:
        try:
            resp = self.session.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout)
        except requests.Timeout as exc:
            raise OllamaTimeoutError(f"Ollama request timed out after {self.timeout:.0f}s") from exc
        except requests.ConnectionError as exc:
            raise OllamaUnavailableError(f"Lost connection to Ollama at {self.host}") from exc

        if resp.status_code == 404:
            raise _NonRetryable(OllamaModelMissingError(
                f"Model '{self.model}' not found. Run: ollama pull {self.model}"))
        if resp.status_code == 429 or resp.status_code >= 500:
            raise OllamaUnavailableError(f"Ollama server error HTTP {resp.status_code}")
        if resp.status_code != 200:
            raise _NonRetryable(OllamaError(f"Ollama rejected the request: HTTP {resp.status_code}"))
        try:
            body = resp.json()
        except ValueError as exc:
            raise LLMOutputError("Ollama returned a non-JSON HTTP body") from exc
        if body.get("error"):
            msg = str(body["error"])
            if "not found" in msg.lower():
                raise _NonRetryable(OllamaModelMissingError(f"{msg}. Run: ollama pull {self.model}"))
            raise _NonRetryable(OllamaError(f"Ollama error: {msg}"))
        text = (body.get("response") or "").strip()
        if not text:
            raise LLMOutputError("The model returned an empty response")
        if body.get("done_reason") == "length":
            log.warning("Model output was cut off by the token limit (num_predict=%d)", self.num_predict)
        return text


class _NonRetryable(OllamaError):
    """Internal wrapper; unwrapped by generate()."""

    def __init__(self, inner: OllamaError) -> None:
        super().__init__(str(inner))
        self.inner = inner
