import pytest

from app.retrieval.reranker import sigmoid


def test_sigmoid_maps_raw_scores_into_zero_to_one():
    assert sigmoid(0) == pytest.approx(0.5)
    assert sigmoid(8.6) > 0.99  # a direct answer
    assert sigmoid(-11.2) < 0.001  # unrelated text


def test_sigmoid_keeps_the_order_of_scores():
    raw = [-11.0, -2.4, 0.66, 5.5, 8.6]
    mapped = [sigmoid(x) for x in raw]
    assert mapped == sorted(mapped)


def test_sigmoid_survives_extreme_inputs():
    assert sigmoid(1e9) == pytest.approx(1.0)
    assert sigmoid(-1e9) == pytest.approx(0.0, abs=1e-12)
