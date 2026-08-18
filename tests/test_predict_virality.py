from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "predict_virality.py"
SPEC = importlib.util.spec_from_file_location("predict_virality", MODULE_PATH)
assert SPEC and SPEC.loader
predict_virality = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(predict_virality)


def _masks() -> dict[str, np.ndarray]:
    mask = np.zeros(predict_virality.EXPECTED_VERTICES, dtype=bool)
    mask[:4] = True
    return {"salience": mask, "valuation": mask, "affective": mask}


def test_opening_window_is_weighted_and_positive_only() -> None:
    predictions = np.zeros((3, predict_virality.EXPECTED_VERTICES), dtype=np.float32)
    predictions[0, :4] = 4.0
    predictions[1, :4] = -20.0  # negative response is not counted as activation
    predictions[2, :4] = 1.0
    times = np.array([0.0, 1.0, 6.0], dtype=np.float32)

    result = predict_virality.score_prediction(
        predictions, times, _masks(), opening_seconds=5.0, decay=0.0
    )

    assert result["opening_segments"] == 2
    assert result["affective_opening_mean"] == pytest.approx(2.0)
    assert result["affective_full_mean"] == pytest.approx(5.0 / 3.0)


def test_batch_ranking_marks_top_candidate() -> None:
    base = {
        "affective_opening_mean": 1.0,
        "affective_opening_peak": 1.0,
        "salience_opening_mean": 1.0,
        "valuation_opening_mean": 1.0,
    }
    results = [
        {"variant": "low", **base},
        {"variant": "high", **{key: value * 3 for key, value in base.items()}},
    ]

    ranked = predict_virality.rank_results(results)

    assert ranked[0]["variant"] == "high"
    assert ranked[0]["virality_proxy"] == pytest.approx(100.0)
    assert ranked[0]["meets_70th_percentile"] is True
    assert ranked[1]["meets_70th_percentile"] is False


def test_reference_pool_keeps_percentile_stable() -> None:
    reference = []
    for value in (1.0, 2.0, 3.0, 4.0):
        reference.append(
            {
                "affective_opening_mean": value,
                "affective_opening_peak": value,
                "salience_opening_mean": value,
                "valuation_opening_mean": value,
            }
        )
    candidates = [
        {
            "variant": "new",
            "affective_opening_mean": 3.5,
            "affective_opening_peak": 3.5,
            "salience_opening_mean": 3.5,
            "valuation_opening_mean": 3.5,
        }
    ]

    ranked = predict_virality.rank_results(candidates, reference)

    assert ranked[0]["score_basis"] == "frozen_reference_pool"
    assert ranked[0]["virality_proxy"] == pytest.approx(75.0)
    assert ranked[0]["meets_70th_percentile"] is True
