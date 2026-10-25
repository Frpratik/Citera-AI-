import json

from tests.pdf_factory import make_pdf

PASSWORD = "correct horse battery"
REFUND_TEXT = "Refund requests must be filed within 14 days of purchase."


def signup(client, email="alice@example.com", password=PASSWORD) -> dict:
    """Register and log in; returns the Authorization header for later requests."""
    assert client.post("/api/auth/register", json={"email": email, "password": password}).status_code == 201
    token = client.post("/api/auth/login", json={"email": email, "password": password}).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def upload(client, headers, name="terms.pdf", pages=(REFUND_TEXT,), process=True):
    """Upload a PDF. With process=True the background worker is run to completion first,
    so the document is already ready (or failed) when the test continues."""
    response = client.post(
        "/api/documents",
        files={"file": (name, make_pdf(list(pages)), "application/pdf")},
        headers=headers,
    )
    if process:
        client.services.worker.drain()
    return response


def parse_sse(text: str) -> list[tuple[str, dict]]:
    """Turn a Server-Sent Events body into [(event_name, data), ...]."""
    events = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.split("\n"))
        events.append((fields["event"], json.loads(fields["data"])))
    return events
