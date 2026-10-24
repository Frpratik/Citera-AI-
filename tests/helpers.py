"""Test helpers: a tiny PDF generator, fake embeddings and a fake LLM.

Fakes let the whole pipeline run offline: no Groq key, no model download.
"""

import math
import re
import zlib

from langchain_core.embeddings import Embeddings

from app.services.llm_service import RagAnswer, RouteDecision
from app.services.retrieval_service import RetrievedChunk


# --------------------------------------------------------------------------- PDF builder
def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(pages: list[str]) -> bytes:
    """Build a minimal valid text PDF with one page per string (no extra libraries needed)."""
    page_count = len(pages)
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(page_count))
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for index, text in enumerate(pages):
        content_id = 5 + 2 * index
        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Contents {content_id} 0 R /Resources << /Font << /F1 3 0 R >> >> >>"
            ).encode()
        )
        # Wrap long text into lines of ~80 characters.
        words, lines, current = text.split(), [], ""
        for word in words:
            if len(current) + len(word) + 1 > 80:
                lines.append(current)
                current = word
            else:
                current = f"{current} {word}".strip()
        if current:
            lines.append(current)
        body = "BT /F1 11 Tf 50 740 Td 14 TL " + " ".join(f"({_escape(l)}) Tj T*" for l in lines) + " ET"
        stream = body.encode("latin-1")
        objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")

    output = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output += f"{number} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref_start = len(output)
    output += f"xref\n0 {len(objects) + 1}\n".encode()
    output += b"0000000000 65535 f \n"
    for offset in offsets:
        output += f"{offset:010d} 00000 n \n".encode()
    output += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n"
    ).encode()
    return bytes(output)


# --------------------------------------------------------------------------- fake embeddings
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "what", "of", "to", "in", "for", "and", "do", "does",
    "be", "must", "can", "at", "on", "by", "how", "when", "who", "i", "it", "that", "this",
}


class FakeEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings: texts sharing words get similar vectors."""

    DIMENSIONS = 512

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.DIMENSIONS
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            if word in _STOPWORDS:
                continue
            vector[zlib.crc32(word.encode()) % self.DIMENSIONS] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


# --------------------------------------------------------------------------- fake LLM
class FakeLLM:
    """Stands in for LLMService. Records calls so tests can assert on them."""

    is_configured = True

    def __init__(self):
        self.rag_calls: list[list[RetrievedChunk]] = []
        self.general_calls: list[str] = []
        self.rag_response: RagAnswer | None = None

    def classify(self, question: str, history: list[dict]) -> RouteDecision:
        q = question.lower().strip(" !.?")
        if q in {"hi", "hello", "hey", "thanks"}:
            return RouteDecision(route="greeting", reason="greeting")
        if "maybe" in q:
            return RouteDecision(route="uncertain", reason="ambiguous")
        if any(word in q for word in ("attendance", "fee", "hostel", "exam", "scholarship")):
            return RouteDecision(route="college_document", reason="college topic")
        return RouteDecision(route="general_knowledge", reason="general")

    def answer_rag(self, question: str, chunks: list[RetrievedChunk]) -> RagAnswer:
        self.rag_calls.append(chunks)
        if self.rag_response is not None:
            return self.rag_response
        return RagAnswer(
            answer="The minimum attendance required is 75% [1].",
            sufficient_evidence=True,
            used_source_ids=[1],
            conflict_detected=False,
        )

    def answer_general(self, question: str, history: list[dict]) -> str:
        self.general_calls.append(question)
        return "This is a general answer."


ATTENDANCE_TEXT = (
    "Attendance Policy. Students must maintain a minimum attendance of 75% in every subject "
    "to be eligible for the semester examinations. Medical leave requires a doctor certificate."
)
FEES_TEXT = (
    "Fee Structure. The annual tuition fee is 85,000 rupees. The examination fee must be paid "
    "before the last date announced by the examination cell."
)
