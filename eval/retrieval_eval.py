"""Retrieval evaluation: how well does the SEARCH find the right chunk? (No chat AI needed.)

    python -m eval.retrieval_eval            # the standard report
    python -m eval.retrieval_eval --chunks   # also compare chunk sizes (slower)

Three questions are answered with numbers:
  1. Which search stage is best?  (meaning only, keywords only, merged, with/without re-ranking)
  2. Where should the relevance cut-off be?  (keep real answers, reject junk)
  3. Which chunk size works best?
"""

import argparse
import json
from pathlib import Path

from eval.dataset import QUESTIONS
from eval.harness import STAGES, EvalEnvironment
from eval.metrics import first_relevant_rank, hit_at, mean, reciprocal_rank

RESULTS_DIR = Path(__file__).parent / "results"
THRESHOLDS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
TOP_K = 5
PIPELINE = "hybrid-wide+rerank"  # the stage that matches what the real app does

ANSWERABLE = [q for q in QUESTIONS if q.answerable]
UNANSWERABLE = [q for q in QUESTIONS if q.expect == "abstain"]


def evaluate_stages(env: EvalEnvironment) -> dict:
    """Per stage: Hit@1/3/5, MRR (overall and by question category), plus raw rerank scores."""
    ranks: dict[str, dict[str, int | None]] = {stage: {} for stage in STAGES}
    all_stages: dict[str, dict] = {}
    for q in QUESTIONS:
        if not (q.answerable or q.expect == "abstain"):
            continue
        results = env.stages(q.search_text)
        all_stages[q.id] = results
        if q.answerable:
            for stage in STAGES:
                ranks[stage][q.id] = first_relevant_rank(results[stage], q.relevant)

    report = {}
    for stage in STAGES:
        by_question = ranks[stage]
        report[stage] = {
            "hit@1": mean(hit_at(r, 1) for r in by_question.values()),
            "hit@3": mean(hit_at(r, 3) for r in by_question.values()),
            "hit@5": mean(hit_at(r, 5) for r in by_question.values()),
            "mrr": mean(reciprocal_rank(r) for r in by_question.values()),
            "by_category_hit@5": {
                cat: mean(hit_at(by_question[q.id], 5) for q in ANSWERABLE if q.category == cat)
                for cat in sorted({q.category for q in ANSWERABLE})
            },
            "missed": [q.id for q in ANSWERABLE if not hit_at(by_question[q.id], 5)],
        }
    return {"stages": report, "raw": all_stages, "ranks": ranks}


def evaluate_threshold(raw: dict) -> list[dict]:
    """For each cut-off: how many real answers survive it, and how many junk searches get through."""
    rows = []
    for t in THRESHOLDS:
        kept = []
        for q in ANSWERABLE:
            top = [h for h in raw[q.id][PIPELINE][:TOP_K] if h.score >= t]
            kept.append(first_relevant_rank(top, q.relevant) is not None)
        passed = [any(h.score >= t for h in raw[q.id][PIPELINE][:TOP_K]) for q in UNANSWERABLE]
        recall, leak = mean(kept), mean(passed)
        rows.append({"threshold": t, "answers_kept": recall, "junk_passed": leak, "gap": recall - leak})
    return rows


RERANK_CUTOFFS = (0.05, 0.1, 0.2, 0.3, 0.5)
SIMILARITY_CUTOFFS = (None, 0.25, 0.3, 0.35, 0.4, 0.45)
JUNK_WEIGHT = 0.5  # missing a real answer is worse than letting junk through: the AI can still say "not found"


def gate_stats(raw: dict, rerank_cutoff: float, similarity_cutoff: float | None) -> tuple[float, float]:
    """(share of answerable questions whose right chunk survives the gate,
        share of unanswerable questions where something still gets through)."""

    def survivors(question):
        similarity = {h.chunk_id: h.score for h in raw[question.id]["dense"]}
        top = raw[question.id][PIPELINE][:TOP_K]
        return [
            h for h in top
            if h.score >= rerank_cutoff
            or (similarity_cutoff is not None and similarity.get(h.chunk_id, -1) >= similarity_cutoff)
        ]

    kept = mean(first_relevant_rank(survivors(q), q.relevant) is not None for q in ANSWERABLE)
    junk = mean(bool(survivors(q)) for q in UNANSWERABLE)
    return kept, junk


def evaluate_gate(raw: dict) -> list[dict]:
    """The two-signal rule: keep a chunk if the re-ranker scores it >= r OR meaning search
    scores it >= s. Tries every pair and scores each by  answers_kept - 0.5 * junk_passed."""
    rows = []
    for r in RERANK_CUTOFFS:
        for s in SIMILARITY_CUTOFFS:
            kept, junk = gate_stats(raw, r, s)
            rows.append({"rerank": r, "similarity": s, "answers_kept": kept, "junk_passed": junk,
                         "score": kept - JUNK_WEIGHT * junk})
    return sorted(rows, key=lambda row: (-round(row["score"], 6), -(row["rerank"]), -(row["similarity"] or 0)))


def print_gate_table(rows: list[dict]) -> None:
    print("\nTWO-SIGNAL RELEVANCE RULE   (keep if re-rank >= r  OR  meaning-similarity >= s)")
    print(f"{'r':>6}{'s':>7}{'answers kept':>15}{'junk through':>14}{'score':>8}")
    for row in rows[:6]:
        s = "none" if row["similarity"] is None else f"{row['similarity']:.2f}"
        print(f"{row['rerank']:>6.2f}{s:>7}{row['answers_kept']:>15.2f}{row['junk_passed']:>14.2f}{row['score']:>8.2f}")
    today = next(r for r in rows if r["rerank"] == 0.2 and r["similarity"] is None)
    print(f"  for comparison, the old rule (r=0.20, no s): kept {today['answers_kept']:.2f}, junk {today['junk_passed']:.2f}")


def best_threshold(rows: list[dict]) -> float:
    """The cut-off with the biggest gap between answers kept and junk let through
    (ties go to the lower, safer-for-recall value)."""
    return max(rows, key=lambda r: (round(r["gap"], 6), -r["threshold"]))["threshold"]


def print_stage_table(report: dict) -> None:
    print(f"\nSEARCH STAGES   (answerable questions, n={len(ANSWERABLE)})")
    print(f"{'stage':<16}{'Hit@1':>8}{'Hit@3':>8}{'Hit@5':>8}{'MRR':>8}")
    for stage, r in report.items():
        print(f"{stage:<16}{r['hit@1']:>8.2f}{r['hit@3']:>8.2f}{r['hit@5']:>8.2f}{r['mrr']:>8.2f}")
    cats = list(report["dense"]["by_category_hit@5"])
    print("\nHit@5 BY QUESTION TYPE")
    print(f"{'stage':<16}" + "".join(f"{c:>12}" for c in cats))
    for stage, r in report.items():
        print(f"{stage:<16}" + "".join(f"{r['by_category_hit@5'][c]:>12.2f}" for c in cats))
    for stage, r in report.items():
        if r["missed"]:
            print(f"  not found in top 5 by {stage}: {', '.join(r['missed'])}")
    print(f"  (the real app uses: {PIPELINE})")


def print_threshold_table(rows: list[dict]) -> None:
    print(f"\nRELEVANCE CUT-OFF   ({PIPELINE}; {len(ANSWERABLE)} answerable, {len(UNANSWERABLE)} unanswerable)")
    print(f"{'cut-off':>8}{'answers kept':>15}{'junk let through':>18}{'gap':>8}")
    for r in rows:
        print(f"{r['threshold']:>8.2f}{r['answers_kept']:>15.2f}{r['junk_passed']:>18.2f}{r['gap']:>8.2f}")
    print(f"  best cut-off on this data: {best_threshold(rows)}")


def run(sweep_chunks: bool = False, haystack_pages: int = 6) -> dict:
    env = EvalEnvironment()
    try:
        out: dict = {}
        configs = [(None, None)]
        if sweep_chunks:
            configs = [(500, 75), (1000, 150), (1500, 225), (2500, 300)]
        for size, overlap in configs:
            settings = env.load(size, overlap, haystack_pages)
            label = f"chunk_size={settings.chunk_size} overlap={settings.chunk_overlap} ({env.chunk_count} chunks)"
            print(f"\n=== {label} ===")
            result = evaluate_stages(env)
            print_stage_table(result["stages"])
            thresholds = evaluate_threshold(result["raw"])
            gate = evaluate_gate(result["raw"])
            if not sweep_chunks or (size, overlap) == (1000, 150):
                print_threshold_table(thresholds)
                print_gate_table(gate)
            out[label] = {"stages": result["stages"], "thresholds": thresholds, "gate": gate[:10]}
        RESULTS_DIR.mkdir(exist_ok=True)
        (RESULTS_DIR / "retrieval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        return out
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", action="store_true", help="also compare several chunk sizes")
    parser.add_argument("--haystack", type=int, default=6, help="pages per distractor document (default 6)")
    args = parser.parse_args()
    run(sweep_chunks=args.chunks, haystack_pages=args.haystack)
