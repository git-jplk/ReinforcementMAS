from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .controller_helper import SequentialControllerFeatures

VALID_STYLES = {"sequential", "mixture", "distillation", "deliberation"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _group_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record.get("sample_id", "unknown_sample")), []).append(record)
    return grouped


def _label_from_future_loss(
    current: Mapping[str, Any],
    following: Optional[Mapping[str, Any]],
    *,
    improvement_threshold: float,
) -> int:
    if bool(current.get("last_round", False)) or following is None:
        return 0
    current_loss = _safe_float(current.get("loss"), 0.0)
    next_loss = _safe_float(following.get("loss"), current_loss)
    return int((current_loss - next_loss) > improvement_threshold)


def collect_from_sequential_trace(
    raw_records: Sequence[Mapping[str, Any]],
    *,
    improvement_threshold: float = 0.01,
) -> List[Dict[str, Any]]:
    """Create controller rows from sequential traces.

    Loss is used only to create an offline training target. It is deliberately excluded
    from the feature vector because gold-token loss is unavailable at inference time.
    """
    rows: List[Dict[str, Any]] = []
    for sample_id, records in _group_records(raw_records).items():
        ordered = sorted(records, key=lambda row: int(row.get("round_idx", 0)))
        for index, record in enumerate(ordered):
            previous = ordered[index - 1] if index > 0 else None
            following = ordered[index + 1] if index + 1 < len(ordered) else None
            row = dict(record)
            row["sample_id"] = sample_id
            row["last_round"] = bool(row.get("last_round", following is None))
            features = SequentialControllerFeatures.from_round(row, previous=previous)
            features["style"] = "sequential"
            features["sample_id"] = sample_id
            features["round_idx"] = int(row.get("round_idx", index))
            features["max_depth"] = int(row.get("max_depth", len(ordered)))
            features["label"] = _label_from_future_loss(
                row,
                following,
                improvement_threshold=improvement_threshold,
            )
            rows.append(features)
    return rows


def build_controller_dataset(
    records: Sequence[Mapping[str, Any]],
    style: Optional[str] = None,
    *,
    improvement_threshold: float = 0.01,
    **_ignored: Any,
) -> List[Dict[str, Any]]:
    selected_style = str(style or "sequential").strip().lower()
    if selected_style not in VALID_STYLES:
        raise ValueError(f"Unsupported style: {selected_style}. Expected one of {sorted(VALID_STYLES)}")
    if selected_style != "sequential":
        raise NotImplementedError("Inference-aligned controller features currently support sequential style only")
    return collect_from_sequential_trace(records, improvement_threshold=improvement_threshold)


def save_controller_dataset(dataset: Sequence[Mapping[str, Any]], path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in dataset:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
