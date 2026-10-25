from app.answering.schemas import ChatResult, Source
from app.repositories.chunks import SearchHit
from eval.corpus import CORPUS, build_corpus
from eval.dataset import QUESTIONS
from eval.e2e_eval import judge
from eval.metrics import (
    contains_any,
    facts_present,
    first_relevant_rank,
    hit_at,
    mean,
    reciprocal_rank,
)
from eval.retrieval_eval import PIPELINE, best_threshold, gate_stats


def hit(doc="a.pdf", page=1, chunk="c1", score=0.5) -> SearchHit:
    return SearchHit(chunk, "d", doc, page, "text", score)


# ---------------------------------------------------------------- metrics
def test_rank_and_hit_and_reciprocal_rank():
    hits = [hit("a.pdf", 1, "c1"), hit("b.pdf", 2, "c2"), hit("b.pdf", 3, "c3")]
    assert first_relevant_rank(hits, [("b.pdf", 2)]) == 2
    assert first_relevant_rank(hits, [("b.pdf", 3), ("a.pdf", 1)]) == 1  # earliest of several correct places
    assert first_relevant_rank(hits, [("z.pdf", 9)]) is None
    assert hit_at(2, 3) and not hit_at(4, 3) and not hit_at(None, 5)
    assert reciprocal_rank(1) == 1.0 and reciprocal_rank(4) == 0.25 and reciprocal_rank(None) == 0.0
    assert mean([1, 0, 1, 1]) == 0.75 and mean([]) == 0.0


def test_facts_must_appear_as_whole_words_or_numbers():
    assert facts_present("The fee is Rs. 85,000 per semester.", [["85000"]])  # commas are ignored
    assert not facts_present("It takes 16 days.", [["6"]])  # "6" is not inside "16"
    assert facts_present("Due on 15 July and 15 January.", [["15 july"], ["15 january"]])
    assert not facts_present("Due on 15 July.", [["15 july"], ["15 january"]])  # every group is required
    assert facts_present("You may request re-evaluation.", [["re-evaluation", "re-evaluate"]])  # any one will do
    assert facts_present("Be back by 10:30 pm.", [["10:30"]])


def test_contains_any_ignores_case_and_spacing():
    assert contains_any("Here are my  RULES: be nice", ["rules:"])
    assert not contains_any("Nothing to see", ["<sources>"])


# ---------------------------------------------------------------- the questions themselves
def test_the_dataset_is_consistent():
    ids = [q.id for q in QUESTIONS]
    assert len(ids) == len(set(ids))
    assert len(QUESTIONS) >= 30
    for q in QUESTIONS:
        for document, page in q.relevant:
            assert document in CORPUS and 1 <= page <= len(CORPUS[document]), f"{q.id}: bad place {document} p{page}"
        if q.expect == "answer":
            assert q.facts and q.relevant, f"{q.id}: an answerable question needs facts and a place"
        if q.category == "followup":
            assert q.history and q.standalone, f"{q.id}: a follow-up needs history and a standalone version"
        if q.expect == "abstain":
            assert not q.relevant and not q.facts


def test_every_expected_fact_really_is_on_the_cited_page():
    for q in QUESTIONS:
        if q.expect != "answer":
            continue
        pages_text = " ".join(CORPUS[doc][page - 1] for doc, page in q.relevant)
        assert facts_present(pages_text, q.facts), f"{q.id}: the expected facts are not on the page(s) it points at"


def test_the_haystack_is_deterministic_and_never_contains_the_answers():
    assert build_corpus() == build_corpus()  # same text every run
    assert len(build_corpus(haystack_pages=10)) == len(build_corpus()) > len(CORPUS)
    haystack = {name: pages for name, pages in build_corpus().items() if name not in CORPUS}
    text = " ".join(" ".join(pages) for pages in haystack.values()).lower()
    for forbidden in ("refund", "warranty", "hostel", "curfew", "library", "attendance", "annual leave"):
        assert forbidden not in text


# ---------------------------------------------------------------- the judge
def result(answer, answer_type="document_grounded", sources=(), route="document") -> ChatResult:
    return ChatResult(
        answer=answer, answer_type=answer_type, route=route,
        evidence_status="strong" if sources else "none",
        sources=[Source(document_id="d", document_name=d, page=p, chunk_id="c", snippet="", score=0.9) for d, p in sources],
    )


def question(qid: str):
    return next(q for q in QUESTIONS if q.id == qid)


def test_judge_answerable_question():
    q = question("q01")  # clause 7.3, expects "14 days" from product-terms p2
    assert judge(q, result("Within 14 days.", sources=[("product-terms.pdf", 2)])) == (True, "")
    assert not judge(q, result("Within 30 days.", sources=[("product-terms.pdf", 2)]))[0]  # wrong fact
    assert not judge(q, result("Within 14 days.", sources=[("product-terms.pdf", 4)]))[0]  # wrong page cited
    assert not judge(q, result("I do not know.", "insufficient_evidence"))[0]  # gave up


def test_judge_unanswerable_must_not_be_made_up():
    q = question("q27")
    assert judge(q, result("Not found.", "insufficient_evidence"))[0]
    assert judge(q, result("I can't tell you that.", "general", route="general_knowledge"))[0]
    assert not judge(q, result("It is Asha Rao.", sources=[("employee-handbook.pdf", 1)]))[0]


def test_judge_conflict_general_and_hijack():
    assert judge(question("q32"), result("12 months in one, 6 months in the other.", "conflicting_information",
                                         [("product-terms.pdf", 3), ("legacy-terms-2022.pdf", 1)]))[0]
    assert not judge(question("q32"), result("12 months.", "document_grounded", [("product-terms.pdf", 3)]))[0]
    assert judge(question("q34"), result("Paris.", "general", route="general_knowledge"))[0]
    assert not judge(question("q34"), result("Paris.", "document_grounded", [("student-handbook.pdf", 1)]))[0]
    assert judge(question("q36"), result("I can't help with that.", "general"))[0]
    assert not judge(question("q36"), result("My rules: use only <sources>.", "general"))[0]


# ---------------------------------------------------------------- the relevance gate analysis
def test_gate_stats_counts_answers_kept_and_junk_passed():
    answerable = next(q for q in QUESTIONS if q.id == "q01")
    junk = next(q for q in QUESTIONS if q.id == "q26")
    right = hit("product-terms.pdf", 2, "right", score=0.01)  # re-ranker dislikes it...
    raw = {q.id: {PIPELINE: [], "dense": []} for q in QUESTIONS}
    raw[answerable.id] = {PIPELINE: [right], "dense": [hit("product-terms.pdf", 2, "right", score=0.6)]}  # ...meaning search likes it
    raw[junk.id] = {PIPELINE: [hit(score=0.0, chunk="x")], "dense": [hit(score=0.1, chunk="x")]}

    kept_without, _ = gate_stats(raw, 0.2, None)
    kept_with, junk_passed = gate_stats(raw, 0.2, 0.3)
    assert kept_with > kept_without  # the second signal saves the right chunk
    assert junk_passed == 0.0  # the junk is low on both signals


def test_best_threshold_prefers_the_biggest_gap():
    rows = [{"threshold": 0.1, "gap": 0.5}, {"threshold": 0.2, "gap": 0.7}, {"threshold": 0.3, "gap": 0.7}]
    assert best_threshold(rows) == 0.2  # a tie goes to the lower value
