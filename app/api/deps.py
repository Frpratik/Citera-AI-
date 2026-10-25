"""Wiring: builds every service once, and the small helpers routes depend on."""

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from psycopg_pool import ConnectionPool

from app.answering.generator import AnswerGenerator, GroqAnswerGenerator
from app.answering.service import AnswerService
from app.api.ratelimit import RateLimiter
from app.auth.passwords import PasswordService
from app.auth.service import AuthService
from app.auth.tokens import TokenService, resolve_jwt_secret
from app.core.config import Settings
from app.core.errors import AuthenticationError
from app.embeddings.service import Embedder, SentenceTransformerEmbedder
from app.jobs.worker import Worker
from app.llm.client import LLMClient
from app.observability.metrics import Metrics
from app.repositories.chunks import ChunkRepository
from app.repositories.documents import DocumentRepository
from app.repositories.jobs import JobRepository
from app.repositories.messages import MessageRepository
from app.repositories.traces import TraceRepository
from app.repositories.users import UserRecord, UserRepository
from app.retrieval.reranker import CrossEncoderReranker, Reranker
from app.retrieval.service import RetrievalService
from app.services.chat_service import ChatService
from app.services.ingestion_service import IngestionService
from app.workflow.graph import ChatWorkflow, build_chat_workflow
from app.workflow.planner import GroqQuestionPlanner, QuestionPlanner
from app.workflow.responder import GeneralResponder, GroqGeneralResponder


@dataclass
class Services:
    settings: Settings
    pool: ConnectionPool
    llm: LLMClient
    auth: AuthService
    ingestion: IngestionService
    chat: ChatService
    workflow: ChatWorkflow  # the same workflow ChatService uses, for tools that pass their own history
    worker: Worker
    metrics: Metrics
    jobs: JobRepository
    traces: TraceRepository
    embedder: Embedder
    reranker: Reranker
    user_limiter: RateLimiter  # per signed-in user: upload and chat
    auth_limiter: RateLimiter  # per client address: register and login


def build_services(
    settings: Settings,
    pool: ConnectionPool,
    *,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    planner: QuestionPlanner | None = None,
    generator: AnswerGenerator | None = None,
    responder: GeneralResponder | None = None,
    passwords: PasswordService | None = None,
) -> Services:
    """Create and connect everything. Tests pass fakes for the AI parts."""
    metrics = Metrics()
    llm = LLMClient(settings, metrics)
    embedder = embedder or SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_batch_size)
    reranker = reranker or CrossEncoderReranker(settings.rerank_model)

    documents = DocumentRepository(pool)
    chunks = ChunkRepository(pool)
    users = UserRepository(pool)
    jobs = JobRepository(pool)
    ingestion = IngestionService(documents, chunks, jobs, embedder, settings)

    retrieval = RetrievalService(
        chunks, embedder, reranker,
        candidates=settings.retrieval_candidates,
        top_k=settings.retrieval_top_k,
        min_score=settings.min_relevance_score,
        min_similarity=settings.min_similarity,
    )
    answerer = AnswerService(generator or GroqAnswerGenerator(llm), settings.strong_relevance_score)
    workflow = build_chat_workflow(
        planner or GroqQuestionPlanner(llm),
        retrieval,
        answerer,
        responder or GroqGeneralResponder(llm),
        documents.count_ready,
        metrics,
    )
    traces = TraceRepository(pool)

    return Services(
        settings=settings,
        pool=pool,
        llm=llm,
        auth=AuthService(
            users,
            passwords or PasswordService(),
            TokenService(resolve_jwt_secret(settings), settings.jwt_expire_minutes),
        ),
        ingestion=ingestion,
        chat=ChatService(workflow, MessageRepository(pool), settings.history_turns, traces, metrics),
        metrics=metrics,
        jobs=jobs,
        traces=traces,
        workflow=workflow,
        worker=Worker(jobs, ingestion, settings),
        embedder=embedder,
        reranker=reranker,
        user_limiter=RateLimiter(settings.rate_limit_per_minute),
        auth_limiter=RateLimiter(settings.auth_rate_limit_per_minute),
    )


# ---------------------------------------------------------------- dependencies for routes
def get_services(request: Request) -> Services:
    return request.app.state.services


ServicesDep = Annotated[Services, Depends(get_services)]

_bearer = HTTPBearer(auto_error=False)  # auto_error off so WE send our standard error shape


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    services: ServicesDep,
) -> UserRecord:
    if credentials is None:
        raise AuthenticationError("Missing bearer token.")
    return services.auth.user_from_token(credentials.credentials)


CurrentUser = Annotated[UserRecord, Depends(get_current_user)]


def rate_limited_user(user: CurrentUser, services: ServicesDep) -> UserRecord:
    """For expensive endpoints: the signed-in user, after the per-user rate limit passes."""
    services.user_limiter.check(user.id)
    return user


RateLimitedUser = Annotated[UserRecord, Depends(rate_limited_user)]


def limit_by_address(request: Request, services: ServicesDep) -> None:
    """For register and login: limit by client address (sign-in is not known yet)."""
    services.auth_limiter.check(request.client.host if request.client else "unknown")
