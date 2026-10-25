import logging
import secrets
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.api import routes_auth, routes_chat, routes_documents, routes_usage
from app.api.deps import Services, build_services
from app.core.config import get_settings
from app.core.errors import AppError, AuthenticationError, NotFoundError
from app.core.logging import request_id_var, setup_logging
from app.db.migrate import run_migrations
from app.db.pool import create_pool
from app.jobs.worker import WorkerLoop

logger = logging.getLogger(__name__)

# A multipart upload is saved to a temporary file by the framework BEFORE our code runs,
# so an oversized upload must be turned away here, from its declared size.
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"

# The browser may only load scripts, styles and data from this site itself. So even if text from
# a PDF or from the AI ever slipped onto the page as HTML, it could not run a script or send
# anything to another site.
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)
DOCS_PATHS = {"/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"}

UPLOAD_PATH = "/api/documents"
UPLOAD_OVERHEAD_BYTES = 64 * 1024  # the multipart wrapping around the file itself


def error_response(status: int, code: str, message: str, details=None, headers=None) -> JSONResponse:
    body = {"error": {"code": code, "message": message}}
    if details:
        body["error"]["details"] = details
    return JSONResponse(status_code=status, content=body, headers=headers)


def _count_request(request: Request, status: int, seconds: float) -> None:
    """Count the request in the metrics, by route PATTERN ("/api/documents/{document_id}"),
    never by the real path, so ids cannot create endless different labels."""
    services = getattr(request.app.state, "services", None)
    if services is None:
        return
    route = getattr(request.scope.get("route"), "path", "unmatched")
    services.metrics.http_requests.labels(request.method, route, str(status)).inc()
    services.metrics.http_seconds.labels(request.method, route).observe(seconds)


def _warm_up(services: Services) -> None:
    try:
        services.embedder.warm_up()
        services.reranker.warm_up()
    except Exception:
        logger.exception("Model warm-up failed")


def create_app(services: Services | None = None) -> FastAPI:
    """Build the app. Tests pass ready-made `services`; normally they are built at startup."""
    settings = services.settings if services else get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging(settings.log_level, settings.log_format)
        pool = None
        if services is None:
            run_migrations(settings.database_url)  # before the pool: it needs the vector type
            pool = create_pool(settings.database_url)
            app.state.services = build_services(settings, pool)
        # Load the AI models in the background now, so the first question does not wait for them.
        threading.Thread(target=_warm_up, args=(app.state.services,), name="warm-up", daemon=True).start()
        worker_loop = None
        if settings.embedded_worker:
            worker_loop = WorkerLoop(app.state.services.worker, settings)
            worker_loop.start()
        logger.info("%s started (env=%s)", settings.app_name, settings.environment)
        yield
        if worker_loop is not None:
            worker_loop.stop()  # let the job in progress finish before the database closes
        if pool is not None:
            pool.close()
        logger.info("shutting down")

    app = FastAPI(title="Citera AI", version="2.0.0", lifespan=lifespan)
    if services is not None:
        app.state.services = services

    # ---------------------------------------------------------------- middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.middleware("http")
    async def request_id_and_safety(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        token = request_id_var.set(request_id)
        started = time.monotonic()
        try:
            length = request.headers.get("content-length", "")
            too_big = (
                request.method == "POST"
                and request.url.path == UPLOAD_PATH
                and length.isdigit()
                and int(length) > settings.max_upload_bytes + UPLOAD_OVERHEAD_BYTES
            )
            if too_big:
                response = error_response(
                    413, "file_too_large", f"File is too large. The limit is {settings.max_upload_mb} MB."
                )
            else:
                response = await call_next(request)
        finally:
            request_id_var.reset(token)
        _count_request(request, response.status_code, time.monotonic() - started)
        response.headers["X-Request-ID"] = request_id
        if request.url.path not in DOCS_PATHS:  # the API docs page needs inline scripts
            response.headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    # ---------------------------------------------------------------- errors
    @app.exception_handler(AppError)
    async def handle_app_error(_request: Request, exc: AppError):
        headers = {"WWW-Authenticate": "Bearer"} if exc.status_code == 401 else None
        return error_response(exc.status_code, exc.code, exc.message, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_request: Request, exc: RequestValidationError):
        details = [
            {"field": ".".join(str(p) for p in e["loc"]), "message": e["msg"]} for e in exc.errors()
        ]
        return error_response(422, "validation_error", "The request was not valid.", details)

    @app.exception_handler(Exception)
    async def handle_unexpected(_request: Request, exc: Exception):
        # Log only the type: exception messages can contain user document text.
        logger.error("Unhandled error: %s", type(exc).__name__, exc_info=exc)
        return error_response(500, "internal_error", "Something went wrong on the server.")

    # ---------------------------------------------------------------- routes
    @app.get("/health", tags=["health"])
    def health():
        """Is the app running? (No database check: this is what a load balancer pings.)"""
        return {"status": "ok"}

    @app.get("/health/ready", tags=["health"])
    def ready(request: Request):
        """Can the app do its job? Checks the database and reports whether the AI is configured."""
        current: Services = request.app.state.services
        try:
            with current.pool.connection() as conn:
                conn.execute("SELECT 1")
        except Exception as exc:  # noqa: BLE001 - any failure here simply means "not ready"
            logger.error("Readiness check failed: %s", type(exc).__name__)
            return error_response(503, "not_ready", "The database is not reachable.")
        return {
            "status": "ready",
            "llm_configured": current.llm.is_configured,
            "llm_circuits": current.llm.circuit_states(),
            "jobs": current.jobs.count_by_status(),
        }

    @app.get("/metrics", include_in_schema=False)
    def metrics(request: Request):
        """Prometheus metrics. Guarded by METRICS_TOKEN if one is set."""
        if not settings.metrics_enabled:
            raise NotFoundError("Not found.")
        if settings.metrics_token:
            supplied = request.headers.get("authorization", "").removeprefix("Bearer ")
            if not secrets.compare_digest(supplied.encode(), settings.metrics_token.encode()):
                raise AuthenticationError("Invalid metrics token.")
        current: Services = request.app.state.services
        for status, count in current.jobs.count_by_status().items():
            current.metrics.jobs.labels(status).set(count)
        body, content_type = current.metrics.render()
        return Response(body, media_type=content_type)

    app.include_router(routes_auth.router)
    app.include_router(routes_documents.router)
    app.include_router(routes_chat.router)
    app.include_router(routes_usage.router)

    # The web page. Mounted LAST so it can never hide an API route above.
    if FRONTEND_DIR.is_dir():
        app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    return app


app = create_app()
