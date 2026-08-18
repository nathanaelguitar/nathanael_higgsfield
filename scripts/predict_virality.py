#!/usr/bin/env python3
"""Rank candidate media by an affective TRIBE activation proxy.

The input for each candidate is an ``.npz`` file containing:

    preds: (n_segments, 20484) TRIBE cortical predictions
    times: (n_segments,) segment start times in seconds
    variant: optional scalar string

This intentionally produces a *relative proxy percentile*, not a probability
of virality.  TRIBE predicts cortical responses and is not trained on social
engagement.  The score emphasizes the first five seconds and affective-
salience/valuation cortical regions available on fsaverage5.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_VERTICES = 20_484  # fsaverage5: 10,242 vertices per hemisphere
DEFAULT_OPENING_SECONDS = 5.0
DEFAULT_DECAY = 0.35

# These are cortical proxies for the circuits described in the campaign
# hypothesis. TRIBE's released fsaverage5 output does not contain subcortical
# amygdala, hypothalamus, hippocampus, brainstem, cerebellar, or autonomic data.
ROI_LABELS: dict[str, tuple[str, ...]] = {
    "salience": (
        "G_Ins_lg_and_S_cent_ins",
        "G_insular_short",
        "S_circular_insula_ant",
        "S_circular_insula_inf",
        "S_circular_insula_sup",
        "G_and_S_cingul-Ant",
        "G_and_S_cingul-Mid-Ant",
    ),
    "valuation": (
        "G_front_inf-Orbital",
        "G_orbital",
        "G_rectus",
        "G_subcallosal",
        "S_orbital_lateral",
        "S_orbital_med-olfact",
        "S_orbital-H_Shaped",
        "Pole_temporal",
        "G_temp_sup-Plan_polar",
    ),
}


def build_roi_masks(data_dir: Path) -> dict[str, np.ndarray]:
    """Fetch Destrieux fsaverage5 labels and return boolean vertex masks."""

    try:
        from nilearn import datasets
    except ImportError as exc:  # pragma: no cover - exercised in setup checks
        raise RuntimeError(
            "nilearn is required for ROI scoring; rerun scripts/setup_tribe.sh"
        ) from exc

    atlas = datasets.fetch_atlas_surf_destrieux(
        data_dir=str(data_dir), verbose=0
    )
    label_to_id = {name: i for i, name in enumerate(atlas.labels)}
    masks: dict[str, np.ndarray] = {}
    for group, labels in ROI_LABELS.items():
        missing = sorted(set(labels) - set(label_to_id))
        if missing:
            raise RuntimeError(f"Atlas is missing ROI labels: {missing}")
        ids = [label_to_id[label] for label in labels]
        left = np.isin(np.asarray(atlas.map_left), ids)
        right = np.isin(np.asarray(atlas.map_right), ids)
        masks[group] = np.concatenate((left, right))

    masks["affective"] = masks["salience"] | masks["valuation"]
    for name, mask in masks.items():
        if mask.shape != (EXPECTED_VERTICES,):
            raise RuntimeError(
                f"ROI {name} has {mask.size} vertices; expected {EXPECTED_VERTICES}"
            )
    return masks


def _weighted_mean(values: np.ndarray, times: np.ndarray, decay: float) -> float:
    if values.size == 0:
        return 0.0
    weights = np.exp(-decay * np.maximum(times, 0.0))
    return float(np.average(values, weights=weights))


def score_prediction(
    predictions: np.ndarray,
    times: np.ndarray,
    roi_masks: dict[str, np.ndarray],
    *,
    opening_seconds: float = DEFAULT_OPENING_SECONDS,
    decay: float = DEFAULT_DECAY,
) -> dict[str, float | int]:
    """Compute opening-window activation features for one candidate.

    Positive predicted response is treated as activation. The combined raw
    score is deliberately simple and interpretable; batch ranking converts it
    to a relative ``virality_proxy`` later.
    """

    predictions = np.asarray(predictions, dtype=np.float32)
    times = np.asarray(times, dtype=np.float32).reshape(-1)
    if predictions.ndim != 2 or predictions.shape[1] != EXPECTED_VERTICES:
        raise ValueError(
            f"preds must have shape (segments, {EXPECTED_VERTICES}), got {predictions.shape}"
        )
    if predictions.shape[0] != times.size:
        raise ValueError("preds and times must contain the same number of segments")
    if opening_seconds <= 0 or decay < 0:
        raise ValueError("opening_seconds must be positive and decay non-negative")

    opening = times < opening_seconds
    if not np.any(opening):
        raise ValueError("candidate has no prediction segment in the opening window")

    activation = np.maximum(predictions, 0.0)
    output: dict[str, float | int] = {
        "opening_segments": int(opening.sum()),
        "opening_seconds": float(opening_seconds),
    }
    for group in ("salience", "valuation", "affective"):
        mask = np.asarray(roi_masks[group], dtype=bool)
        if mask.shape != (EXPECTED_VERTICES,) or not mask.any():
            raise ValueError(f"invalid or empty ROI mask: {group}")
        per_segment = activation[:, mask].mean(axis=1)
        opening_values = per_segment[opening]
        opening_times = times[opening]
        output[f"{group}_opening_mean"] = _weighted_mean(
            opening_values, opening_times, decay
        )
        output[f"{group}_opening_peak"] = float(opening_values.max())
        output[f"{group}_full_mean"] = float(per_segment.mean())

    # Mean activation is the main term; peak, salience, and valuation add
    # interpretable emphasis without pretending these weights are learned.
    output["raw_activation_score"] = float(
        0.55 * output["affective_opening_mean"]
        + 0.20 * output["affective_opening_peak"]
        + 0.15 * output["salience_opening_mean"]
        + 0.10 * output["valuation_opening_mean"]
    )
    return output


def _rank_percentiles(values: np.ndarray) -> np.ndarray:
    """Return stable 0-100 rank percentiles; one-item batches are indeterminate."""

    if values.size == 1:
        return np.array([50.0], dtype=np.float32)
    order = np.argsort(np.argsort(values, kind="stable"), kind="stable")
    return (100.0 * order / (values.size - 1)).astype(np.float32)


def _empirical_percentiles(
    values: np.ndarray, reference_values: np.ndarray
) -> np.ndarray:
    """Map values to percentiles against a frozen reference distribution."""

    if reference_values.size == 0:
        raise ValueError("reference distribution cannot be empty")
    ordered = np.sort(reference_values.astype(float))
    return (
        100.0 * np.searchsorted(ordered, values, side="right") / ordered.size
    ).astype(np.float32)


def rank_results(
    results: list[dict[str, Any]],
    reference_results: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Add a relative or frozen-reference activation percentile."""

    if not results:
        return results
    component_names = (
        "affective_opening_mean",
        "affective_opening_peak",
        "salience_opening_mean",
        "valuation_opening_mean",
    )
    if reference_results:
        percentiles = {
            name: _empirical_percentiles(
                np.asarray([r[name] for r in results], dtype=float),
                np.asarray([r[name] for r in reference_results], dtype=float),
            )
            for name in component_names
        }
        basis = "frozen_reference_pool"
    else:
        percentiles = {
            name: _rank_percentiles(np.asarray([r[name] for r in results], dtype=float))
            for name in component_names
        }
        basis = "current_candidate_batch"
    for i, result in enumerate(results):
        for name, values in percentiles.items():
            result[f"{name}_percentile"] = round(float(values[i]), 2)
        result["virality_proxy"] = round(
            float(
                0.55 * percentiles["affective_opening_mean"][i]
                + 0.20 * percentiles["affective_opening_peak"][i]
                + 0.15 * percentiles["salience_opening_mean"][i]
                + 0.10 * percentiles["valuation_opening_mean"][i]
            ),
            2,
        )
        result["meets_70th_percentile"] = result["virality_proxy"] >= 70.0
        result["score_basis"] = basis
        result["confidence"] = "low-until-calibrated"
    return sorted(results, key=lambda item: item["virality_proxy"], reverse=True)


def _load_prediction(path: Path) -> tuple[str, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        if "preds" not in data or "times" not in data:
            raise ValueError(f"{path} must contain preds and times arrays")
        variant = path.stem
        if "variant" in data:
            variant = str(np.asarray(data["variant"]).reshape(()).item())
        return variant, np.asarray(data["preds"]), np.asarray(data["times"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", nargs="+", type=Path, help="TRIBE .npz files")
    parser.add_argument(
        "--reference",
        nargs="+",
        type=Path,
        help="frozen TRIBE .npz reference pool for stable cross-run percentiles",
    )
    parser.add_argument("--output", type=Path, help="write ranked JSON to this path")
    parser.add_argument(
        "--atlas-data-dir",
        type=Path,
        default=Path.home() / ".cache" / "nilearn_data",
    )
    parser.add_argument("--opening-seconds", type=float, default=DEFAULT_OPENING_SECONDS)
    parser.add_argument("--decay", type=float, default=DEFAULT_DECAY)
    args = parser.parse_args()

    masks = build_roi_masks(args.atlas_data_dir)

    def score_paths(paths: list[Path]) -> list[dict[str, Any]]:
        scored: list[dict[str, Any]] = []
        for path in paths:
            variant, predictions, times = _load_prediction(path)
            result = score_prediction(
                predictions,
                times,
                masks,
                opening_seconds=args.opening_seconds,
                decay=args.decay,
            )
            result["variant"] = variant
            result["source"] = str(path)
            scored.append(result)
        return scored

    results = score_paths(args.predictions)
    reference_results = score_paths(args.reference) if args.reference else None

    payload = {
        "results": rank_results(results, reference_results),
        "reference_pool_size": len(reference_results) if reference_results else None,
        "interpretation": {
            "score": (
                "empirical percentile against the frozen reference pool"
                if reference_results
                else "relative percentile within the supplied candidate batch"
            ),
            "not": "a calibrated probability of virality or a percentage of the brain",
            "directly_measured": ["cortical salience proxy", "cortical valuation proxy"],
            "not_directly_measured": [
                "amygdala",
                "hypothalamus",
                "hippocampus",
                "brainstem",
                "cerebellum",
            ],
        },
    }
    rendered = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
