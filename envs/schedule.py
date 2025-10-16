"""Reward weight scheduling utilities shared across curriculum configs.

See `envs/geospot.py` for usage.

"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

StageWeights = Dict[str, float]
ScheduleEntry = Dict[str, Any]


def _weights_for_stage(stage: str, target_weights: StageWeights) -> StageWeights:
    """Return default field weights for a curriculum stage keyword."""
    key = stage.lower()
    if key == "country":
        return {
            "country": target_weights["country"],
            "region": 0.0,
            "city": 0.0,
        }
    if key == "region":
        return {
            "country": target_weights["country"],
            "region": target_weights["region"],
            "city": 0.0,
        }
    # city/coords/full all map to the full set of weights
    return {
        "country": target_weights["country"],
        "region": target_weights["region"],
        "city": target_weights["city"],
    }


def prepare_weight_schedule(
    schedule: Optional[Mapping[int, Any]],
    target_weights: StageWeights,
) -> Dict[int, ScheduleEntry]:
    """Normalize a curriculum schedule into sorted thresholds.

    Inputs may be strings (aliasing predefined stages) or dictionaries with
    ``label``/``weights`` keys. The result is always ``{step: {label, weights}}``
    sorted by step to make downstream lookups deterministic.
    """

    # Default curriculum schedule: country -> region -> city
    default_schedule: Dict[int, Any] = {0: "country", 1500: "region", 3000: "city"}
    raw_schedule = dict(schedule) if schedule is not None else default_schedule

    normalized: Dict[int, ScheduleEntry] = {}

    for raw_threshold, raw_value in raw_schedule.items():
        try:
            threshold = int(raw_threshold)
        except (TypeError, ValueError):
            continue

        label = "custom"
        weights: StageWeights | None = None

        if isinstance(raw_value, str):
            label = raw_value.lower()
            weights = _weights_for_stage(raw_value, target_weights)
        elif isinstance(raw_value, Mapping):
            maybe_label = raw_value.get("label")
            if isinstance(maybe_label, str):
                label = maybe_label.lower()
            raw_weights = raw_value.get("weights", raw_value)
            if isinstance(raw_weights, Mapping):
                weights = {
                    "country": float(raw_weights.get("country", target_weights["country"])),
                    "region": float(raw_weights.get("region", 0.0)),
                    "city": float(raw_weights.get("city", 0.0)),
                }

        if weights is None:
            continue

        normalized[threshold] = {"label": label, "weights": dict(weights)}

    if not normalized:
        normalized[0] = {"label": "full", "weights": dict(target_weights)}

    return dict(sorted(normalized.items()))
