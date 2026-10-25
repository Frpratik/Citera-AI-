import logging
import secrets
from datetime import UTC, datetime, timedelta

import jwt

from app.core.config import Settings
from app.core.errors import AuthenticationError

logger = logging.getLogger(__name__)

ALGORITHM = "HS256"


def resolve_jwt_secret(settings: Settings) -> str:
    """The signing secret. Missing is tolerated only in dev, never in production."""
    if settings.jwt_secret:
        return settings.jwt_secret
    if settings.environment == "dev":
        logger.warning("JWT_SECRET is not set: using a random one; logins end when the app restarts.")
        return secrets.token_urlsafe(48)
    raise RuntimeError("JWT_SECRET must be set when ENVIRONMENT is not 'dev'.")


class TokenService:
    """Issues and checks login tokens (JWT). A token says "this is user X until time T",
    and carries a signature only our secret can produce, so it cannot be edited."""

    def __init__(self, secret: str, expire_minutes: int):
        self._secret = secret
        self._expire = timedelta(minutes=expire_minutes)

    def issue(self, user_id: str) -> str:
        now = datetime.now(UTC)
        payload = {"sub": user_id, "iat": now, "exp": now + self._expire}
        return jwt.encode(payload, self._secret, algorithm=ALGORITHM)

    def verify(self, token: str) -> str:
        """Return the user id, or raise AuthenticationError (bad, forged or expired)."""
        try:
            payload = jwt.decode(
                token,
                self._secret,
                algorithms=[ALGORITHM],  # pinned: a token claiming "no signature" is rejected
                options={"require": ["exp", "sub"]},
            )
        except jwt.PyJWTError as exc:
            raise AuthenticationError("Invalid or expired token.") from exc
        return str(payload["sub"])
