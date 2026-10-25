"""Configuration: defaults, environment variables and optional .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse

from .errors import ConfigError

WHISPER_MODEL = "small"
GEMMA_MODEL = "gemma3:4b"
OLLAMA_HOST = "http://localhost:11434"
CHUNK_SIZE_WORDS = 1200
CHUNK_OVERLAP_WORDS = 100
TEMPERATURE = 0.2

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


@dataclass(frozen=True)
class Config:
    whisper_model: str = WHISPER_MODEL
    llm_model: str = GEMMA_MODEL
    ollama_host: str = OLLAMA_HOST
    chunk_size_words: int = CHUNK_SIZE_WORDS
    chunk_overlap_words: int = CHUNK_OVERLAP_WORDS
    temperature: float = TEMPERATURE
    language: str | None = None
    device: str = "auto"            # auto | cpu | cuda
    compute_type: str = "auto"      # auto -> int8 on CPU, float16 on CUDA
    vad_filter: bool = True
    output_dir: Path = Path("notes_out")
    request_timeout: float = 600.0  # seconds per Ollama request
    max_retries: int = 3            # retries after the first attempt
    num_ctx: int = 8192             # context window requested from Ollama
    num_predict: int = 4096         # max generated tokens per request
    merge_max_chars: int = 14000    # max characters of summaries per merge request
    json_attempts: int = 3          # attempts to obtain valid JSON per chunk
    lecture_date: date | None = None
    verbose: bool = False

    @property
    def host_is_local(self) -> bool:
        return (urlparse(self.ollama_host).hostname or "") in LOCAL_HOSTS

    def validate(self) -> "Config":
        if self.chunk_size_words < 50:
            raise ConfigError("chunk size must be at least 50 words")
        if self.chunk_overlap_words < 0:
            raise ConfigError("chunk overlap cannot be negative")
        if self.chunk_overlap_words >= self.chunk_size_words:
            raise ConfigError("chunk overlap must be smaller than the chunk size")
        if not 0.0 <= self.temperature <= 2.0:
            raise ConfigError("temperature must be between 0 and 2")
        parsed = urlparse(self.ollama_host)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ConfigError(f"invalid Ollama host: {self.ollama_host!r} (expected http://host:port)")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ConfigError("device must be one of: auto, cpu, cuda")
        if self.max_retries < 0 or self.json_attempts < 1:
            raise ConfigError("retry counts must be non-negative (json attempts >= 1)")
        return self

    def with_overrides(self, **kw: object) -> "Config":
        return replace(self, **{k: v for k, v in kw.items() if v is not None})  # type: ignore[arg-type]


def load_dotenv(path: Path = Path(".env")) -> dict[str, str]:
    """Tiny KEY=VALUE parser (no dependency). Real environment variables win."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        values[key.strip()] = val.strip().strip("'\"")
    return values


def _num(env: Mapping[str, str], key: str, cast, default):
    raw = env.get(key)
    if raw in (None, ""):
        return default
    try:
        return cast(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be a valid {cast.__name__}, got {raw!r}") from exc


def config_from_env(env: Mapping[str, str] | None = None, dotenv: Path | None = Path(".env")) -> Config:
    merged: dict[str, str] = {}
    if dotenv is not None:
        merged.update(load_dotenv(dotenv))
    merged.update(os.environ if env is None else env)
    return Config(
        whisper_model=merged.get("WHISPER_MODEL") or WHISPER_MODEL,
        llm_model=merged.get("GEMMA_MODEL") or GEMMA_MODEL,
        ollama_host=(merged.get("OLLAMA_HOST") or OLLAMA_HOST).rstrip("/"),
        chunk_size_words=_num(merged, "CHUNK_SIZE_WORDS", int, CHUNK_SIZE_WORDS),
        chunk_overlap_words=_num(merged, "CHUNK_OVERLAP_WORDS", int, CHUNK_OVERLAP_WORDS),
        temperature=_num(merged, "TEMPERATURE", float, TEMPERATURE),
        device=merged.get("WHISPER_DEVICE") or "auto",
        compute_type=merged.get("WHISPER_COMPUTE_TYPE") or "auto",
        output_dir=Path(merged.get("OUTPUT_DIR") or "notes_out"),
        request_timeout=_num(merged, "OLLAMA_TIMEOUT", float, 600.0),
        max_retries=_num(merged, "OLLAMA_MAX_RETRIES", int, 3),
        num_ctx=_num(merged, "OLLAMA_NUM_CTX", int, 8192),
        num_predict=_num(merged, "OLLAMA_NUM_PREDICT", int, 4096),
    )
