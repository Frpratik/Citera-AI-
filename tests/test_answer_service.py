from app.answering.schemas import RagAnswer
from app.answering.service import (
    NO_EVIDENCE_MESSAGE,
    PASSAGES_ONLY_MESSAGE,
    UNVERIFIED_CITATION_MESSAGE,
    AnswerService,
)
from app.core.errors import LLMUnavailableError
from app.repositories.chunks import SearchHit

CHUNKS = [
    SearchHit("c1", "d1", "terms.pdf", 3, "Clause 7.3: a refund request must be filed within 14 days.", 0.99),
    SearchHit("c2", "d1", "terms.pdf", 4, "Clause 7.4: a refund may be denied if the product is used.", 0.55),
]


class FakeGenerator:
    """Returns a scripted draft and remembers whether it was called."""

    def __init__(self, draft: RagAnswer | None = None):
        self.draft = draft
        self.calls = 0

    def generate(self, question, chunks):
        self.calls += 1
        return self.draft


def service(draft: RagAnswer | None = None) -> tuple[AnswerService, FakeGenerator]:
    generator = FakeGenerator(draft)
    return AnswerService(generator, strong_score=0.8), generator


def test_no_chunks_means_the_ai_is_never_called():
    svc, generator = service()
    result = svc.answer("anything", [])
    assert generator.calls == 0
    assert result.answer == NO_EVIDENCE_MESSAGE
    assert result.answer_type == "insufficient_evidence" and result.evidence_status == "none"


def test_when_the_ai_is_down_the_matching_passages_are_shown_instead_of_an_error():
    generator = FakeGenerator()
    generator.generate = lambda question, chunks: (_ for _ in ()).throw(LLMUnavailableError("down"))
    result = AnswerService(generator, strong_score=0.8).answer("deadline?", CHUNKS)

    assert result.answer_type == "passages_only"
    assert PASSAGES_ONLY_MESSAGE in result.answer
    assert "terms.pdf, page 3" in result.answer and "terms.pdf, page 4" in result.answer  # each with its place
    assert [(s.document_name, s.page) for s in result.sources] == [("terms.pdf", 3), ("terms.pdf", 4)]
    assert result.evidence_status == "strong"  # from the search scores, not from an AI


def test_a_good_answer_is_grounded_with_sources_from_our_data():
    draft = RagAnswer(answer="File within 14 days [1].", sufficient_evidence=True, used_source_ids=[1])
    result = service(draft)[0].answer("deadline?", CHUNKS)
    assert result.answer_type == "document_grounded"
    assert result.evidence_status == "strong"
    assert [(s.document_name, s.page) for s in result.sources] == [("terms.pdf", 3)]
    assert result.warnings == []


def test_moderate_evidence_when_only_a_weaker_chunk_is_cited():
    draft = RagAnswer(answer="It may be denied [2].", sufficient_evidence=True, used_source_ids=[2])
    result = service(draft)[0].answer("denied?", CHUNKS)
    assert result.evidence_status == "moderate"


def test_the_ai_saying_it_does_not_know_is_passed_on_without_sources():
    draft = RagAnswer(answer="The documents don't mention the fee.", sufficient_evidence=False)
    result = service(draft)[0].answer("fee?", CHUNKS)
    assert result.answer_type == "insufficient_evidence"
    assert result.answer == "The documents don't mention the fee."
    assert result.sources == []


def test_invented_citations_are_rejected():
    draft = RagAnswer(answer="Sure, it is 14 days [9].", sufficient_evidence=True, used_source_ids=[9])
    result = service(draft)[0].answer("deadline?", CHUNKS)
    assert result.answer == UNVERIFIED_CITATION_MESSAGE
    assert result.answer_type == "insufficient_evidence" and result.sources == []


def test_a_confident_answer_without_any_citation_is_rejected():
    draft = RagAnswer(answer="It is 14 days.", sufficient_evidence=True, used_source_ids=[])
    assert service(draft)[0].answer("deadline?", CHUNKS).answer == UNVERIFIED_CITATION_MESSAGE


def test_figures_missing_from_the_cited_text_produce_a_warning():
    draft = RagAnswer(answer="Within 30 days [1].", sufficient_evidence=True, used_source_ids=[1])
    result = service(draft)[0].answer("deadline?", CHUNKS)
    assert result.answer_type == "document_grounded"  # still answered, but flagged
    assert "30" in result.warnings[0]


def test_conflicting_sources_are_flagged():
    draft = RagAnswer(
        answer="One says 14 days [1], another says it may be denied [2].",
        sufficient_evidence=True, used_source_ids=[1, 2], conflict_detected=True,
    )
    result = service(draft)[0].answer("refund?", CHUNKS)
    assert result.answer_type == "conflicting_information"
    assert len(result.sources) == 2
