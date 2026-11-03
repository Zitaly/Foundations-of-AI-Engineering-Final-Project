"""Language model clients. The rest of PawPal+ only uses the LLM protocol, so tests can swap in a fake model.

Free-tier friendly: replies are cached (in memory and on disk), requests are paced to a per-model
requests-per-minute limit, and a busy, rate-limited, retired, or timed-out model hands off to the next
model. FallbackLLM chains providers (Gemini, then Groq); when every provider fails, the app falls back
to its built-in rules.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any, Protocol

DEFAULT_MODEL = "gemini-3.1-flash-lite"  # override with the GEMINI_MODEL environment variable
FALLBACK_MODELS = ["gemini-3.5-flash-lite", "gemini-3.5-flash"]  # each has its own free-tier quota
GROQ_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]  # first can be overridden with GROQ_MODEL
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
BUSY_CODES = {413, 429, 500, 503, 504}  # too large for this model's per-minute budget, rate-limited, overloaded, timed out
MISSING_CODES = {404}  # model retired or misspelled; skip it for the rest of the session
MAX_WAIT_SECONDS = 60  # longest we'll pause when every model is busy
DEFAULT_RPM = 15  # Gemini requests per minute per model; override with GEMINI_RPM to match your quota
GROQ_RPM = 30  # Groq requests per minute per model; override with GROQ_RPM
MAX_OUTPUT_TOKENS = 1024  # every PawPal+ reply is short; capping it keeps token-per-minute use down
GROQ_MAX_TOKENS = 2048  # Groq's models reason before answering, and reasoning counts toward this
CACHE_SIZE = 128
CACHE_DIR = Path(os.environ.get("PAWPAL_CACHE_DIR", Path(__file__).parent / "data" / "ai_cache"))


def retry_delay(error: Exception) -> float:
    """Seconds to leave a busy model alone: the delay the API asks for, or a default."""
    headers = getattr(error, "headers", None)
    try:
        if headers is not None and headers.get("retry-after"):
            return float(headers.get("retry-after"))
    except ValueError:
        pass
    match = re.search(r"retry in ([\d.]+)s", str(error))
    return float(match.group(1)) if match else 60.0 if getattr(error, "code", None) == 429 else 10.0


def is_timeout(error: Exception) -> bool:
    """True for request timeouts, which the SDKs raise without an HTTP status code."""
    return isinstance(error, TimeoutError) or "timeout" in type(error).__name__.lower()


class LLMError(RuntimeError):
    """The model couldn't be reached or returned something unusable."""


class APIError(Exception):
    """An HTTP error from a provider, with the status code and headers the retry logic looks at."""

    def __init__(self, code: int, message: str, headers: Any = None):
        super().__init__(f"HTTP {code}: {message}")
        self.code, self.headers = code, headers


class LLM(Protocol):
    """What PawPal+ needs from a language model."""

    def generate_text(self, system: str, prompt: str) -> str: ...

    def generate_json(self, system: str, prompt: str) -> Any: ...


def get_api_key() -> str | None:
    """Return the Gemini API key from the environment, or None if it isn't set."""
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def get_groq_key() -> str | None:
    """Return the Groq API key from the environment, or None if it isn't set."""
    return os.environ.get("GROQ_API_KEY")


def parse_json(text: str) -> Any:
    """Parse a JSON reply, tolerating a ```json fence around it."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise LLMError(f"model returned invalid JSON: {e}") from None


class HostedLLM:
    """Shared caching, pacing, and model fallback for an API provider. Subclasses send one request in _call()."""

    provider = "LLM"

    def __init__(self, models: list[str], rpm: int, temperature: float, cache_dir: Path | None):
        self.model = models[0]
        self.models = models
        self.temperature = temperature
        self.rpm = rpm
        self.max_wait = MAX_WAIT_SECONDS  # set to 0 when another provider can answer instead of waiting
        self.request_count = 0  # requests actually sent, for the sidebar
        self.cache_hits = 0
        self._cache: OrderedDict[tuple[str, str, str], str] = OrderedDict()
        self._cache_path = cache_dir / f"{self.provider.lower()}.json" if cache_dir else None
        self._busy_until: dict[str, float] = {}  # model -> time.monotonic() when it may be tried again
        self._sent: dict[str, deque[float]] = {m: deque() for m in self.models}  # send times in the last minute
        self._load_cache()

    @property
    def name(self) -> str:
        return f"{self.provider} ({self.model})"

    def _call(self, model: str, system: str, prompt: str, mime_type: str) -> str:
        """Send one request to `model` and return the reply text ("" if it was empty or blocked)."""
        raise NotImplementedError

    def _load_cache(self) -> None:
        """Restore replies saved by earlier runs. A damaged cache file is ignored."""
        if not self._cache_path or not self._cache_path.exists():
            return
        try:
            for mime_type, system, prompt, text in json.loads(self._cache_path.read_text(encoding="utf-8"))[-CACHE_SIZE:]:
                self._cache[(mime_type, system, prompt)] = text
        except (OSError, ValueError, TypeError):
            self._cache.clear()

    def _save_cache(self) -> None:
        """Write the cache to disk (best effort: a failed write only loses the cache)."""
        if not self._cache_path:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            temp = self._cache_path.with_suffix(".tmp")
            temp.write_text(json.dumps([[*key, text] for key, text in self._cache.items()]), encoding="utf-8")
            os.replace(temp, self._cache_path)
        except OSError:
            pass

    def _forget(self, key: tuple[str, str, str]) -> None:
        if self._cache.pop(key, None) is not None:
            self._save_cache()

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
        self._save_cache()
        return text

    def _request(self, system: str, prompt: str, mime_type: str) -> str:
        """Send the request to the first model that's ready, waiting briefly if none is."""
        blocked: set[str] = set()  # models that sent an empty reply to this request; asking again won't help
        last_error = ""
        for _ in range(2):  # second pass only after waiting for the first model to free up
            for model in self.models:
                if model in blocked or self._ready_at(model, time.monotonic()) > time.monotonic():
                    continue
                self._sent[model].append(time.monotonic())
                self.request_count += 1
                try:
                    text = self._call(model, system, prompt, mime_type)
                except Exception as e:  # the SDKs raise several error types; surface them all the same way
                    code = getattr(e, "code", None)
                    if code in MISSING_CODES:
                        self._busy_until[model] = math.inf
                    elif code in BUSY_CODES or is_timeout(e):
                        self._busy_until[model] = time.monotonic() + retry_delay(e)
                    else:
                        raise LLMError(f"{self.provider} request failed: {e}") from e
                    last_error = f"{model}: {e}"
                    continue
                if text:
                    return text
                blocked.add(model)
                last_error = f"{model} returned an empty or unusable response"
            ready = [self._ready_at(m, time.monotonic()) for m in self.models if m not in blocked]
            wait = min(ready, default=math.inf) - time.monotonic()
            if wait > self.max_wait:
                break
            time.sleep(max(wait, 0))
        raise LLMError(f"No {self.provider} model could answer right now (last problem: {last_error}).")

    def generate_text(self, system: str, prompt: str) -> str:
        """Return a plain-text reply."""
        return self._generate(system, prompt, "text/plain").strip()

    def generate_json(self, system: str, prompt: str) -> Any:
        """Return a reply parsed as JSON. An invalid reply is dropped from the cache and asked for once more."""
        try:
            return parse_json(self._generate(system, prompt, "application/json"))
        except LLMError as e:
            self._forget(("application/json", system, prompt))
            repair = f"{prompt}\n\nYour previous reply was not valid JSON ({e}). Reply again with only the JSON."
            try:
                return parse_json(self._generate(system, repair, "application/json"))
            except LLMError:
                self._forget(("application/json", system, repair))
                raise


class GeminiLLM(HostedLLM):
    """LLM backed by the Gemini API (google-genai SDK)."""

    provider = "Gemini"

    def __init__(self, api_key: str | None = None, model: str | None = None, temperature: float = 0.2,
                 cache_dir: Path | None = CACHE_DIR):
        key = api_key or get_api_key()
        if not key:
            raise LLMError("No Gemini API key. Set GEMINI_API_KEY or add it to .streamlit/secrets.toml.")
        from google import genai  # imported here so the app and tests still run without the package
        from google.genai import types

        # One attempt per request with a 60 s timeout; _request handles busy models by switching models instead.
        http_options = types.HttpOptions(timeout=60_000, retry_options=types.HttpRetryOptions(attempts=1))
        self._client = genai.Client(api_key=key, http_options=http_options)
        first = model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
        super().__init__(
            [first, *(m for m in FALLBACK_MODELS if m != first)],
            int(os.environ.get("GEMINI_RPM", DEFAULT_RPM)), temperature, cache_dir,
        )

    def _call(self, model: str, system: str, prompt: str, mime_type: str) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            response_mime_type=mime_type,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
        return self._client.models.generate_content(model=model, contents=prompt, config=config).text or ""


class GroqLLM(HostedLLM):
    """LLM backed by Groq's OpenAI-compatible chat completions API, called with the standard library."""

    provider = "Groq"

    def __init__(self, api_key: str | None = None, model: str | None = None, temperature: float = 0.2,
                 cache_dir: Path | None = CACHE_DIR, url: str = GROQ_URL):
        self._key = api_key or get_groq_key()
        if not self._key:
            raise LLMError("No Groq API key. Set GROQ_API_KEY or add it to .streamlit/secrets.toml.")
        self.url = url
        first = model or os.environ.get("GROQ_MODEL", GROQ_MODELS[0])
        super().__init__(
            [first, *(m for m in GROQ_MODELS if m != first)], int(os.environ.get("GROQ_RPM", GROQ_RPM)), temperature, cache_dir,
        )

    def _post(self, body: dict) -> dict:
        """POST `body` as JSON and return the decoded reply. Raises APIError for HTTP errors."""
        request = urllib.request.Request(
            self.url, data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
                "User-Agent": "PawPal+/1.0",  # Groq's Cloudflare front rejects urllib's default agent (error 1010)
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as e:
            raise APIError(e.code, e.read().decode("utf-8", errors="replace")[:300], e.headers) from None
        except urllib.error.URLError as e:
            if isinstance(e.reason, TimeoutError):
                raise TimeoutError(str(e.reason)) from None
            raise

    def _call(self, model: str, system: str, prompt: str, mime_type: str) -> str:
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_completion_tokens": GROQ_MAX_TOKENS,
        }
        if mime_type == "application/json":
            body["response_format"] = {"type": "json_object"}
        if model.startswith("openai/gpt-oss"):
            body["reasoning_effort"] = "low"  # these replies are simple; less reasoning saves tokens and time
        try:
            data = self._post(body)
        except APIError as e:
            if e.code == 400 and "json_validate_failed" in str(e):
                return ""  # the model broke JSON mode; treat like an empty reply and try the next model
            raise
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()  # in case reasoning leaks into the reply


class FallbackLLM:
    """Tries each provider in order and raises LLMError only if every one fails."""

    def __init__(self, providers: list[HostedLLM]):
        if not providers:
            raise ValueError("FallbackLLM needs at least one provider")
        self.providers = providers
        self.last_used: str | None = None  # name of the provider that gave the latest answer

    @property
    def name(self) -> str:
        return " → ".join(p.provider for p in self.providers)

    def _first_answer(self, method: str, system: str, prompt: str) -> Any:
        problems = []
        for provider in self.providers:
            try:
                result = getattr(provider, method)(system, prompt)
            except LLMError as e:
                problems.append(str(e))
                continue
            self.last_used = provider.name
            return result
        raise LLMError(" | ".join(problems))

    def generate_text(self, system: str, prompt: str) -> str:
        return self._first_answer("generate_text", system, prompt)

    def generate_json(self, system: str, prompt: str) -> Any:
        return self._first_answer("generate_json", system, prompt)
