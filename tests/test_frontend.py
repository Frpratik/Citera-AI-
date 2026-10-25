import re
from pathlib import Path

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"


def test_the_page_and_its_files_are_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    assert "<title>Citera AI</title>" in page.text
    assert client.get("/app.js").status_code == 200
    assert "javascript" in client.get("/app.js").headers["content-type"]
    assert client.get("/style.css").status_code == 200


def test_the_api_still_works_next_to_the_page(client):
    assert client.get("/health").json() == {"status": "ok"}  # not swallowed by the page
    assert client.get("/api/documents").status_code == 401  # a real API answer, not a page


def test_a_strict_content_security_policy_is_sent_with_the_page_and_the_api(client):
    for path in ("/", "/health"):
        policy = client.get(path).headers["content-security-policy"]
        assert "script-src 'self'" in policy and "'unsafe-inline'" not in policy
        assert "frame-ancestors 'none'" in policy and "object-src 'none'" in policy


def test_the_api_docs_page_is_exempt_because_it_needs_inline_scripts(client):
    assert "content-security-policy" not in client.get("/docs").headers


def test_the_page_has_no_inline_scripts_styles_or_event_handlers():
    """Those would be blocked by the policy above, and are a way for injected text to run."""
    html = (FRONTEND / "index.html").read_text(encoding="utf-8")
    assert not re.search(r"<script(?![^>]*\bsrc=)", html)  # every script has a src
    assert not re.search(r"<style", html)
    assert not re.search(r"\sstyle\s*=", html)
    assert not re.search(r"\son[a-z]+\s*=", html)  # onclick=, onload=, ...


def test_the_script_never_turns_server_text_into_html():
    source = (FRONTEND / "app.js").read_text(encoding="utf-8")
    js = re.sub(r"//.*", "", source)  # look at the code, not at the comments that explain the rules
    for dangerous in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
        assert dangerous not in js, f"app.js uses {dangerous}"
    assert not re.search(r"\blocalStorage\s*[.\[]", js)  # the login token must not outlive the tab
