import uuid
from datetime import UTC, datetime, timedelta

import jwt

from app.auth.tokens import TokenService
from tests.api_helpers import PASSWORD, signup


def test_register_login_and_me(client):
    body = client.post("/api/auth/register", json={"email": " Alice@Example.com ", "password": PASSWORD})
    assert body.status_code == 201
    assert body.json()["email"] == "alice@example.com"  # tidied
    assert "password" not in body.text and "hash" not in body.text

    headers = signup(client, "bob@example.com")
    me = client.get("/api/auth/me", headers=headers)
    assert me.status_code == 200 and me.json()["email"] == "bob@example.com"


def test_the_password_is_stored_only_as_a_hash(client):
    client.post("/api/auth/register", json={"email": "alice@example.com", "password": PASSWORD})
    with client.services.pool.connection() as conn:
        stored = conn.execute("SELECT password_hash FROM users").fetchone()["password_hash"]
    assert PASSWORD not in stored and stored.startswith("$argon2")


def test_the_same_email_cannot_register_twice_even_in_other_case(client):
    client.post("/api/auth/register", json={"email": "alice@example.com", "password": PASSWORD})
    again = client.post("/api/auth/register", json={"email": "ALICE@example.com", "password": PASSWORD})
    assert again.status_code == 409 and again.json()["error"]["code"] == "email_taken"


def test_wrong_password_and_unknown_email_look_exactly_the_same(client):
    client.post("/api/auth/register", json={"email": "alice@example.com", "password": PASSWORD})
    wrong = client.post("/api/auth/login", json={"email": "alice@example.com", "password": "not the one"})
    unknown = client.post("/api/auth/login", json={"email": "nobody@example.com", "password": PASSWORD})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()


def test_weak_or_malformed_signups_are_rejected(client):
    assert client.post("/api/auth/register", json={"email": "a@b.co", "password": "short"}).status_code == 422
    assert client.post("/api/auth/register", json={"email": "not-an-email", "password": PASSWORD}).status_code == 422


def test_protected_routes_need_a_valid_token(client):
    missing = client.get("/api/documents")
    assert missing.status_code == 401
    assert missing.headers["www-authenticate"] == "Bearer"
    assert client.get("/api/documents", headers={"Authorization": "Bearer garbage"}).status_code == 401


def _token(client, payload_user=None, secret=None, expire_minutes=60):
    user_id = payload_user or str(uuid.uuid4())
    secret = secret or client.services.settings.jwt_secret
    return {"Authorization": f"Bearer {TokenService(secret, expire_minutes).issue(user_id)}"}


def test_expired_forged_and_orphan_tokens_are_rejected(client):
    headers = signup(client)
    real_id = client.get("/api/auth/me", headers=headers).json()["id"]

    assert client.get("/api/auth/me", headers=_token(client, real_id, expire_minutes=-1)).status_code == 401
    assert client.get("/api/auth/me", headers=_token(client, real_id, secret="y" * 48)).status_code == 401
    assert client.get("/api/auth/me", headers=_token(client)).status_code == 401  # no such user
    assert client.get("/api/auth/me", headers=_token(client, "not-a-uuid")).status_code == 401


def test_a_token_that_claims_to_need_no_signature_is_rejected(client):
    headers = signup(client)
    user_id = client.get("/api/auth/me", headers=headers).json()["id"]
    payload = {"sub": user_id, "exp": datetime.now(UTC) + timedelta(hours=1)}
    unsigned = jwt.encode(payload, "", algorithm="none")
    assert client.get("/api/auth/me", headers={"Authorization": f"Bearer {unsigned}"}).status_code == 401


def test_login_attempts_are_rate_limited_by_address(make_client):
    client = make_client(auth_rate_limit_per_minute=3)
    codes = [
        client.post("/api/auth/login", json={"email": "a@b.co", "password": "whatever"}).status_code
        for _ in range(4)
    ]
    assert codes == [401, 401, 401, 429]
