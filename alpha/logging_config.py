"""Structured logging setup.

Library code emits records through ``logging`` only; ``print`` is reserved for
``scripts/``. Handlers are configured by entry points, never at import time.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Final

_RESERVED: Final[frozenset[str]] = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__
) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Render records as one JSON object per line.

    Any non-standard attribute attached to the record (via ``extra=``) is
    merged into the object, so call sites can log structured fields without a
    bespoke formatter per module.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED:
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure(level: int = logging.INFO, *, stream: Any = None) -> None:
    """Install the JSON formatter on the root logger. Idempotent."""
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Module-level logger accessor; adds no handlers."""
    return logging.getLogger(name)
