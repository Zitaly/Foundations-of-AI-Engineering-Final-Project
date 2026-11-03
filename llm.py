"""Gemini wrapper. The rest of PawPal+ only uses the LLM protocol, so tests can swap in a fake model."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Protocol

DEFAULT_MODEL = "gemini-3.8-flash"  # override with the GEMINI_MODEL environment variable
FALLBACK_MODELS = ["gemini-3.7-flash", "gemini-3.5-flash"]  # tried in order when a model is busy
BUSY_CODES = {429, 500, 503, 504}  # rate-limited, overloaded, or timed out
MAX_WAIT_SECONDS = 60  # longest we'll pause when every model is busy


def retry_delay(error: Exception) -> float:
    """Seconds to leave a busy model alone: the delay the API asks for, or a default."""
    match = re.search(r"retry in ([\d.]+)s", str(error))
    return float(match.group(1)) if match else 60.0 if getattr(error, "code", None) == 429 else 10.0


class LLMError(RuntimeError):
    """The model couldn't be reached or returned something unusable."""


class LLM(Protocol):
    """What PawPal+ needs from a language model."""

    def generate_text(self, system: str, prompt: str) -> str: ...

    def generate_json(self, system: str, prompt: str) -> Any: ...


def get_api_key() -> str | None:
    """Return the Gemini API key from the environment, or None if it isn't set."""
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def parse_json(text: str) -> Any:
    """Parse a JSON reply, tolerating a ```json fence around it."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise LLMError(f"model returned invalid JSON: {e}") from None


class GeminiLLM:
    """LLM backed by the Gemini API (google-genai SDK)."""

    def __init__(self, api_key: str | None = None, model: str | None = None, temperature: float = 0.2):
        key = api_key or get_api_key()
        if not key:
            raise LLMError("No Gemini API key. Set GEMINI_API_KEY or add it to .streamlit/secrets.toml.")
        from google import genai  # imported here so the app and tests still run without the package
        from google.genai import types

        # One attempt per request with a 60 s timeout; _generate handles busy models by switching models instead.
        http_options = types.HttpOptions(timeout=60_000, retry_options=types.HttpRetryOptions(attempts=1))
        self._client = genai.Client(api_key=key, http_options=http_options)
        self.model = model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
        self.temperature = temperature
        self._busy_until: dict[str, float] = {}  # model -> time.monotonic() when it may be tried again

    def _generate(self, system: str, prompt: str, mime_type: str) -> str:
        """Send one request and return the reply text."""
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system, temperature=self.temperature, response_mime_type=mime_type
        )
        models = [self.model, *(m for m in FALLBACK_MODELS if m != self.model)]
        for _ in range(2):  # second pass only after waiting out the shortest cooldown
            for model in models:
                if time.monotonic() < self._busy_until.get(model, 0):
                    continue
                try:
                    text = self._client.models.generate_content(model=model, contents=prompt, config=config).text
                except Exception as e:  # the SDK raises several error types; surface them all the same way
                    if getattr(e, "code", None) not in BUSY_CODES:
                        raise LLMError(f"Gemini request failed: {e}") from e
                    self._busy_until[model] = time.monotonic() + retry_delay(e)
                    continue
                if not text:
                    raise LLMError("Gemini returned an empty response (it may have been blocked)")
                return text
            wait = min(self._busy_until[m] for m in models) - time.monotonic()
            if wait > MAX_WAIT_SECONDS:
                break
            time.sleep(max(wait, 0))
        raise LLMError("Every Gemini model is busy or over the free-tier rate limit. Try again in a minute.")

    def generate_text(self, system: str, prompt: str) -> str:
        """Return a plain-text reply."""
        return self._generate(system, prompt, "text/plain").strip()

    def generate_json(self, system: str, prompt: str) -> Any:
        """Return a reply parsed as JSON."""
        return parse_json(self._generate(system, prompt, "application/json"))
