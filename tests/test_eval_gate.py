"""The quality gate: fails when a change makes search measurably worse.

Needs the real AI models and the database, so it is skipped unless you ask for it:

    set RUN_EVAL=1        (PowerShell:  $env:RUN_EVAL = "1")
    pytest tests/test_eval_gate.py

The limits below were set from the measured baseline (see eval/results/retrieval.json) with
a safety margin, so normal noise does not fail the build but a real regression does.
"""

import os

import pytest

from app.core.config import get_settings
from eval.retrieval_eval import PIPELINE, evaluate_stages, gate_stats

pytestmark = [
    pytest.mark.eval,
    pytest.mark.skipif(not os.getenv("RUN_EVAL"), reason="set RUN_EVAL=1 to run the quality gate"),
]


@pytest.fixture(scope="module")
def measured():
    from eval.harness import EvalEnvironment

    env = EvalEnvironment()
    try:
        env.load()
        yield evaluate_stages(env)
    finally:
        env.close()


def test_the_right_chunk_is_found_near_the_top(measured):
    stage = measured["stages"][PIPELINE]
    assert stage["hit@3"] >= 0.92, f"right chunk not in the top 3 often enough: {stage}"
    assert stage["mrr"] >= 0.88, f"right chunk ranked too low on average: {stage}"


def test_re_ranking_still_earns_its_place(measured):
    plain, reranked = measured["stages"]["hybrid"], measured["stages"][PIPELINE]
    assert reranked["mrr"] >= plain["mrr"] + 0.03


def test_the_relevance_gate_keeps_nearly_every_real_answer(measured):
    settings = get_settings()
    kept, junk = gate_stats(measured["raw"], settings.min_relevance_score, settings.min_similarity)
    assert kept >= 0.92, f"the gate throws away real answers: kept {kept:.0%}"
    assert junk <= 0.5, f"the gate lets too much junk through: {junk:.0%}"
