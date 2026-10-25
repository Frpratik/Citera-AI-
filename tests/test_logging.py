import io
import json
import logging

import pytest

from app.core.logging import (
    JsonFormatter,
    RequestIdFilter,
    TextFormatter,
    request_id_var,
)

SECRET = "the confidential sentence from a user's document"


def logger_with(formatter) -> tuple[logging.Logger, io.StringIO]:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(formatter)
    logger = logging.getLogger(f"test-{id(stream)}")
    logger.handlers, logger.propagate = [handler], False
    logger.setLevel(logging.INFO)
    return logger, stream


def test_json_logs_are_one_object_per_line_with_the_request_id_and_extra_fields():
    logger, stream = logger_with(JsonFormatter())
    token = request_id_var.set("req-123")
    try:
        logger.info("chat_trace", extra={"trace": {"total_ms": 840, "llm_calls": 2}})
    finally:
        request_id_var.reset(token)

    line = json.loads(stream.getvalue())
    assert line["message"] == "chat_trace" and line["level"] == "INFO"
    assert line["request_id"] == "req-123"
    assert line["trace"] == {"total_ms": 840, "llm_calls": 2}


@pytest.mark.parametrize("formatter", [TextFormatter(), JsonFormatter()], ids=["text", "json"])
def test_an_exception_shows_where_and_what_type_but_never_its_message(formatter):
    logger, stream = logger_with(formatter)
    try:
        raise ValueError(SECRET)
    except ValueError:
        logger.exception("something failed")

    output = stream.getvalue()
    assert SECRET not in output  # user text must not reach the logs
    assert "ValueError" in output and "test_logging.py" in output  # but we can still debug
