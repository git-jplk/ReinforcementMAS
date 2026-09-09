from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence


FEATURE_NAMES = [
    "round_fraction",
    "planner_to_refiner_norm",
    "refiner_to_solver_norm",
    "feedback_to_planner_norm",
    "planner_norm_delta",
    "refiner_norm_delta",
    "feedback_norm_delta",
    "latent_drift",
    "feedback_cosine_to_previous",
    "is_first_round",
    "is_last_round",
]


class SequentialControllerFeatures:
    """Build controller features available during both training and inference."""

    names = FEATURE_NAMES

    @staticmethod
    def _float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def from_round(
        cls,
        row: Mapping[str, Any],
        *,
        previous: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, float]:
        round_idx = cls._float(row.get("round_idx"))
        max_depth = max(cls._float(row.get("max_depth"), 1.0), 1.0)
        planner_norm = cls._float(row.get("planner_to_refiner_norm"))
        refiner_norm = cls._float(row.get("refiner_to_solver_norm"))
        feedback_norm = cls._float(row.get("feedback_to_planner_norm"))
        previous = previous or {}
        previous_planner = cls._float(previous.get("planner_to_refiner_norm"))
        previous_refiner = cls._float(previous.get("refiner_to_solver_norm"))
        previous_feedback = cls._float(previous.get("feedback_to_planner_norm"))

        return {
            "round_fraction": round_idx / max_depth,
            "planner_to_refiner_norm": planner_norm,
            "refiner_to_solver_norm": refiner_norm,
            "feedback_to_planner_norm": feedback_norm,
            "planner_norm_delta": planner_norm - previous_planner,
            "refiner_norm_delta": refiner_norm - previous_refiner,
            "feedback_norm_delta": feedback_norm - previous_feedback,
            "latent_drift": abs(refiner_norm - planner_norm),
            "feedback_cosine_to_previous": cls._float(row.get("feedback_cosine_to_previous")),
            "is_first_round": float(round_idx <= 0),
            "is_last_round": float(bool(row.get("last_round", False))),
        }

    @classmethod
    def vector(
        cls,
        row: Mapping[str, Any],
        *,
        previous: Optional[Mapping[str, Any]] = None,
    ) -> List[float]:
        features = cls.from_round(row, previous=previous)
        return [features[name] for name in cls.names]
