import pytest

from app.auth.passwords import PasswordService
from app.auth.tokens import TokenService, resolve_jwt_secret
from app.core.config import Settings
from app.core.errors import AuthenticationError
from app.repositories.messages import MessageRepository
from app.repositories.users import UserRepository

SECRET = "s" * 48


def test_passwords_are_hashed_salted_and_verified():
    service = PasswordService()
    first, second = service.hash("my password"), service.hash("my password")
    assert first != "my password" and first != second  # salted: same input, different hash
    assert service.verify(first, "my password") is True
    assert service.verify(first, "other") is False
    assert service.verify("not a real hash", "my password") is False


def test_tokens_roundtrip_and_reject_tampering():
    tokens = TokenService(SECRET, 60)
    token = tokens.issue("user-1")
    assert tokens.verify(token) == "user-1"
    with pytest.raises(AuthenticationError):
        tokens.verify(token[:-3] + "abc")
    with pytest.raises(AuthenticationError):
        TokenService("t" * 48, 60).verify(token)  # signed with a different secret


def test_secret_rules():
    assert resolve_jwt_secret(Settings(jwt_secret="mine", _env_file=None)) == "mine"
    random_a = resolve_jwt_secret(Settings(environment="dev", _env_file=None))
    random_b = resolve_jwt_secret(Settings(environment="dev", _env_file=None))
    assert random_a and random_a != random_b  # dev: random per run
    with pytest.raises(RuntimeError, match="JWT_SECRET"):
        resolve_jwt_secret(Settings(environment="production", _env_file=None))


def test_users_repository(db_pool):
    users = UserRepository(db_pool)
    created = users.create("alice@example.com", "hash-1")
    assert users.get(created.id).email == "alice@example.com"
    assert users.get("not-a-uuid") is None
    user, stored_hash = users.get_credentials("ALICE@example.com")
    assert user.id == created.id and stored_hash == "hash-1"
    assert users.get_credentials("nobody@example.com") is None


def test_messages_repository(db_pool):
    messages = MessageRepository(db_pool)
    for i in range(3):
        messages.add_exchange("alice-owner", f"q{i}", f"a{i}", "general")
    messages.add_exchange("bob-owner", "bob q", "bob a", "general")

    recent = messages.recent("alice-owner", 4)
    assert [m.content for m in recent] == ["q1", "a1", "q2", "a2"]  # latest 4, oldest first
    assert [m.role for m in recent] == ["user", "assistant", "user", "assistant"]
    assert messages.clear("alice-owner") == 6
    assert messages.recent("alice-owner", 10) == []
    assert len(messages.recent("bob-owner", 10)) == 2  # untouched
