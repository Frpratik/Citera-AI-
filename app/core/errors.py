"""Application errors.

Services raise these; `main.py` converts them into one consistent JSON shape:

    {"error": {"code": "document_not_found", "message": "..."}}
"""


class AppError(Exception):
    """Base class for errors whose message is safe to show to the user."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, *, status_code: int | None = None, code: str | None = None):
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        if code is not None:
            self.code = code


class MissingClientIdError(AppError):
    status_code = 400
    code = "missing_client_id"


class InvalidUploadError(AppError):
    status_code = 400
    code = "invalid_upload"


class FileTooLargeError(AppError):
    status_code = 413
    code = "file_too_large"


class DocumentLimitError(AppError):
    status_code = 409
    code = "document_limit_reached"


class DocumentNotFoundError(AppError):
    status_code = 404
    code = "document_not_found"


class RateLimitExceededError(AppError):
    status_code = 429
    code = "rate_limited"


class LLMUnavailableError(AppError):
    status_code = 503
    code = "llm_unavailable"


class RetrievalUnavailableError(AppError):
    status_code = 503
    code = "retrieval_unavailable"
