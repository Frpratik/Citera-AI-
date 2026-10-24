"""FastAPI application entry point.

Run with:  uvicorn app.main:app --reload
Then open: http://localhost:8000
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api import routes_chat, routes_documents
from app.api.deps import Services, get_services
from app.core.config import get_settings
from app.core.errors import AppError
from app.core.logging import setup_logging

logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"


def error_response(status_code: int, code: str, message: str, details=None) -> JSONResponse:
    body: dict = {"error": {"code": code, "message": message}}
    if details:
        body["error"]["details"] = details
    return JSONResponse(status_code=status_code, content=body)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    services = get_services()
    services.documents.recover_after_restart()
    if not services.llm.is_configured:
        logger.warning("GROQ_API_KEY is not set: chat will return an error until it is configured.")
    logger.info("CollegeMate AI started")
    yield


app = FastAPI(title="CollegeMate AI", version="1.0.0", lifespan=lifespan)

# CORS: only the origins listed in settings. The bundled frontend is served from the
# same origin, so this matters only if you host the frontend elsewhere.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-Client-ID"],
)


# --------------------------------------------------------------------------- errors
@app.exception_handler(AppError)
async def handle_app_error(_request: Request, exc: AppError):
    return error_response(exc.status_code, exc.code, exc.message)


@app.exception_handler(RequestValidationError)
async def handle_validation_error(_request: Request, exc: RequestValidationError):
    details = [
        {"field": ".".join(str(part) for part in error["loc"]), "message": error["msg"]}
        for error in exc.errors()
    ]
    return error_response(422, "validation_error", "The request was not valid.", details)


@app.exception_handler(StarletteHTTPException)
async def handle_http_exception(_request: Request, exc: StarletteHTTPException):
    return error_response(exc.status_code, "http_error", str(exc.detail))


@app.exception_handler(Exception)
async def handle_unexpected_error(_request: Request, exc: Exception):
    # Log the type only; messages can contain document text.
    logger.error("Unhandled error: %s", type(exc).__name__, exc_info=True)
    return error_response(500, "internal_error", "Something went wrong on the server.")


# --------------------------------------------------------------------------- routes
@app.get("/health", tags=["health"])
def health(services: Services = Depends(get_services)):
    return {"status": "ok", "llm_configured": services.llm.is_configured}


app.include_router(routes_documents.router)
app.include_router(routes_chat.router)

# The frontend is mounted LAST so it never shadows the API routes above.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
