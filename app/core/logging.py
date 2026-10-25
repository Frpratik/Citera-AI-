import contextvars
import json
import logging
import traceback

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

HIDDEN_MESSAGE = "<message hidden: it can contain user text>"

# Everything a LogRecord has by itself. Anything else on a record was added with `extra=`.
_STANDARD_FIELDS = set(vars(logging.LogRecord("x", 0, "x", 0, "", (), None))) | {
    "message", "asctime", "request_id",
}


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class _SafeExceptions:
    """Shows WHERE an error happened (the stack) and its TYPE, but never its message.

    An exception message can contain what the user uploaded or typed (for example a parser
    quoting the text it choked on), and logs are kept and shared far more widely than data.
    Because this lives in the formatter, it covers every `exc_info=` and `logger.exception`.
    """

    def formatException(self, ei) -> str:
        error_type, _value, tb = ei
        frames = "".join(traceback.format_tb(tb))
        return f"Traceback (most recent call last):\n{frames}{error_type.__name__}: {HIDDEN_MESSAGE}"


class TextFormatter(_SafeExceptions, logging.Formatter):
    def __init__(self):
        super().__init__("%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s")


class JsonFormatter(_SafeExceptions, logging.Formatter):
    """One JSON object per line: easy for log tools to search and filter. Anything passed
    with `extra={...}` becomes its own field."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        payload.update(
            {key: value for key, value in vars(record).items() if key not in _STANDARD_FIELDS}
        )
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str, log_format: str = "text") -> None:
    handler = logging.StreamHandler()
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(JsonFormatter() if log_format == "json" else TextFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
