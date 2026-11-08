"""Logging setup: a rotating file under data/logs, warnings on the console, and the latest lines kept in memory
for the app's sidebar. API keys are scrubbed from every record before it is written anywhere.

Modules log through loggers named "pawpal.<module>"; nothing is written until setup_logging() is called.
"""

from __future__ import annotations

import logging
import os
import re
from collections import deque
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(os.environ.get("PAWPAL_LOG_DIR", Path(__file__).parent / "data" / "logs"))
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
SECRET_PATTERN = re.compile(r"AIza[0-9A-Za-z_\-]{20,}|gsk_[0-9A-Za-z]{20,}|Bearer\s+[^\s\"']+")  # Gemini, Groq, auth headers


def redact(text: str) -> str:
    """Replace anything shaped like an API key with [redacted]."""
    return SECRET_PATTERN.sub("[redacted]", text)


class RedactFilter(logging.Filter):
    """Scrubs API keys from a record's message and exception text."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg, record.args = redact(record.getMessage()), None
        if record.exc_info:
            record.exc_text = redact(logging.Formatter().formatException(record.exc_info))
            record.exc_info = None
        return True


class RecentHandler(logging.Handler):
    """Keeps the last `size` formatted records in memory."""

    def __init__(self, size: int = 50):
        super().__init__()
        self.lines: deque[str] = deque(maxlen=size)

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


recent = RecentHandler()
recent.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))


def setup_logging(level: str | None = None, log_dir: Path | None = LOG_DIR, console: bool = True) -> logging.Logger:
    """Configure the "pawpal" logger once (later calls do nothing) and return it.

    The level comes from `level`, else PAWPAL_LOG_LEVEL, else INFO. Pass log_dir=None to skip the file, and
    console=False to keep warnings off the terminal (the CLI shows them in its own output).
    """
    root = logging.getLogger("pawpal")
    if recent in root.handlers:
        return root
    root.setLevel((level or os.environ.get("PAWPAL_LOG_LEVEL", "INFO")).upper())
    root.propagate = False  # keep PawPal+ lines out of Streamlit's own log output
    stderr = logging.StreamHandler()
    stderr.setLevel(logging.WARNING)
    handlers: list[logging.Handler] = [stderr, recent] if console else [recent]
    if log_dir is not None:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            handlers.append(RotatingFileHandler(log_dir / "pawpal.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"))
        except OSError as e:
            stderr.handle(logging.makeLogRecord({"msg": f"File logging disabled: {e}", "levelno": logging.WARNING, "levelname": "WARNING"}))
    for handler in handlers:
        handler.addFilter(RedactFilter())
        if handler is not recent:
            handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(handler)
    return root
