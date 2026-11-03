"""Gemini wrapper. The rest of PawPal+ only uses the LLM protocol, so tests can swap in a fake model.

Free-tier friendly: replies are cached, requests are paced to a per-model requests-per-minute limit,
and a busy or rate-limited model hands off to a fallback model instead of failing.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import OrderedDict, deque
from typing import Any, Protocol

DEFAULT_MODEL = "gemini-3.1-flash-lite"  # override with the GEMINI_MODEL environment variable
FALLBACK_MODELS = ["gemini-3.5-flash-lite", "gemini-3.5-flash"]  # each has its own free-tier quota
BUSY_CODES = {429, 500, 503, 504}  # rate-limited, overloaded, or timed out
MAX_WAIT_SECONDS = 60  # longest we'll pause when every model is busy
DEFAULT_RPM = 15  # requests per minute per model; override with GEMINI_RPM to match your quota
MAX_OUTPUT_TOKENS = 1024  # every PawPal+ reply is short; capping it keeps token-per-minute use down
CACHE_SIZE = 128


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
        self.models = [self.model, *(m for m in FALLBACK_MODELS if m != self.model)]
        self.temperature = temperature
        self.rpm = int(os.environ.get("GEMINI_RPM", DEFAULT_RPM))
        self.request_count = 0  # requests actually sent, for the sidebar
        self.cache_hits = 0
        self._cache: OrderedDict[tuple[str, str, str], str] = OrderedDict()
        self._busy_until: dict[str, float] = {}  # model -> time.monotonic() when it may be tried again
        self._sent: dict[str, deque[float]] = {m: deque() for m in self.models}  # send times in the last minute

    def _ready_at(self, model: str, now: float) -> float:
        """When `model` may next be called: after any cooldown and once it's under the per-minute limit."""
        sent = self._sent[model]
        while sent and now - sent[0] >= 60:
            sent.popleft()
        paced = sent[0] + 60 if len(sent) >= self.rpm else 0
        return max(self._busy_until.get(model, 0), paced)

    def _generate(self, system: str, prompt: str, mime_type: str) -> str:
        """Return the reply text, from the cache if this exact request was answered before."""
        key = (mime_type, system, prompt)
        if key in self._cache:
            self._cache.move_to_end(key)
            self.cache_hits += 1
            return self._cache[key]
        text = self._request(system, prompt, mime_type)
        self._cache[key] = text
        if len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        return text

    def _request(self, system: str, prompt: str, mime_type: str) -> str:
        """Send the request to the first model that's ready, waiting briefly if none is."""
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            response_mime_type=mime_type,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
        for _ in range(2):  # second pass only after waiting for the first model to free up
            for model in self.models:
                if self._ready_at(model, time.monotonic()) > time.monotonic():
                    continue
                self._sent[model].append(time.monotonic())
                self.request_count += 1
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
            wait = min(self._ready_at(m, time.monotonic()) for m in self.models) - time.monotonic()
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
