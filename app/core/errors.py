class AppError(Exception):
    """Errors whose message is safe to show to the client."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class ValidationFailedError(AppError):
    status_code = 400
    code = "invalid_request"


class InvalidUploadError(AppError):
    status_code = 400
    code = "invalid_upload"


class FileTooLargeError(AppError):
    status_code = 413
    code = "file_too_large"


class AuthenticationError(AppError):
    status_code = 401
    code = "unauthorized"


class EmailTakenError(AppError):
    status_code = 409
    code = "email_taken"


class RateLimitExceededError(AppError):
    status_code = 429
    code = "rate_limited"


class LLMUnavailableError(AppError):
    status_code = 503
    code = "llm_unavailable"


class PDFProcessingError(AppError):
    status_code = 422
    code = "pdf_processing_failed"
