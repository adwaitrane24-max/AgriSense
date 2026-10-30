"""Tests for top-k, threshold and label logic using fake probabilities."""

import numpy as np
import pytest

from leafcheck import config
from leafcheck.predictor import FakePredictor, build_result, pretty_label, top_k

LABELS = list(config.CLASS_NAMES)


def probs_with(best_index: int, best: float) -> list[float]:
    """19 probabilities: `best` at best_index, the rest shared evenly."""
    rest = (1.0 - best) / (len(LABELS) - 1)
    values = [rest] * len(LABELS)
    values[best_index] = best
    return values


def test_top_k_returns_three_sorted_indices() -> None:
    probs = [0.05, 0.4, 0.1, 0.3, 0.15]
    assert top_k(probs, 3) == [1, 3, 4]


def test_top_k_when_k_larger_than_n() -> None:
    assert top_k([0.2, 0.8], 3) == [1, 0]


def test_result_top3_sorted_and_length_three() -> None:
    probs = np.random.default_rng(0).dirichlet(np.ones(len(LABELS)))
    result = build_result(probs, LABELS)
    confidences = [item["confidence"] for item in result["top"]]
    assert len(result["top"]) == 3
    assert confidences == sorted(confidences, reverse=True)
    assert confidences[0] == pytest.approx(max(probs), abs=1e-4)


def test_below_threshold_is_not_sure() -> None:
    result = build_result(probs_with(10, 0.45), LABELS)
    assert result["status"] == "not_sure"
    assert result["prediction"] is None
    assert result["message"] == config.NOT_SURE_MESSAGE


def test_above_threshold_names_the_class() -> None:
    result = build_result(probs_with(10, 0.92), LABELS)
    assert result["status"] == "ok"
    assert result["prediction"]["label"] == "Tomato___Early_blight"
    assert result["prediction"]["condition"] == "Early blight"


def test_exactly_at_threshold_counts_as_confident() -> None:
    result = build_result(probs_with(0, 0.60), LABELS)
    assert result["status"] == "ok"


def test_every_result_has_disclaimer() -> None:
    for best in (0.3, 0.99):
        assert build_result(probs_with(3, best), LABELS)["disclaimer"] == config.DISCLAIMER


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError):
        build_result([0.5, 0.5], LABELS)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Tomato___Early_blight", ("Tomato", "Early blight")),
        ("Tomato___Tomato_mosaic_virus", ("Tomato", "Mosaic virus")),
        ("Pepper,_bell___healthy", ("Bell pepper", "Healthy")),
        ("Corn___Common_rust", ("Corn (maize)", "Common rust")),
    ],
)
def test_pretty_label(raw: str, expected: tuple[str, str]) -> None:
    assert pretty_label(raw) == expected


def test_healthy_flag() -> None:
    result = build_result(probs_with(18, 0.9), LABELS)  # Tomato___healthy
    assert result["prediction"]["healthy"] is True


def test_fake_predictor_is_repeatable_and_sums_to_one() -> None:
    fake = FakePredictor()
    batch = np.full((1, 224, 224, 3), 0.25, dtype=np.float32)
    first, second = fake.predict_proba(batch), fake.predict_proba(batch)
    assert np.allclose(first, second)
    assert first.sum() == pytest.approx(1.0)
    assert fake.calls == 2
