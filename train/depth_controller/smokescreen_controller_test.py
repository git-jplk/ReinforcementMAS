from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List

from controller_helper import FEATURE_NAMES


MAX_ENTRIES = 1000


def parse_args(argv=None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Generate a small labeled controller JSONL smoke dataset.")
	parser.add_argument(
		"--output",
		default="smokescreen_controller_dataset.jsonl",
		help="Output JSONL path",
	)
	parser.add_argument(
		"--entries",
		type=int,
		default=1000,
		help=f"Number of records to generate, capped at {MAX_ENTRIES}",
	)
	parser.add_argument("--seed", type=int, default=42)
	return parser.parse_args(argv)


def _round_record(
	rng: random.Random,
	sample_index: int,
	round_index: int,
	max_depth: int,
	previous: Dict[str, float],
) -> Dict[str, float | int | str | bool]:
	progress = round_index / max_depth
	planner_norm = max(0.05, previous["planner_to_refiner_norm"] + rng.uniform(-0.12, 0.18))
	refiner_norm = max(0.05, planner_norm + rng.uniform(-0.16, 0.22))
	feedback_norm = max(0.05, refiner_norm + rng.uniform(-0.2, 0.2))
	cosine = max(-1.0, min(1.0, previous["feedback_cosine_to_previous"] + rng.uniform(-0.15, 0.15)))
	is_last = round_index == max_depth - 1

	# Continue when the current state is still changing and confidence is not stable.
	improvement_signal = (
		0.55 * (1.0 - progress)
		+ 0.25 * min(abs(feedback_norm - refiner_norm), 1.0)
		+ 0.20 * (1.0 - max(cosine, 0.0))
	)
	label = int(not is_last and improvement_signal > 0.34)

	record: Dict[str, float | int | str | bool] = {
		"style": "sequential",
		"sample_id": f"smoke_sample_{sample_index:04d}",
		"round_idx": round_index,
		"max_depth": max_depth,
		"round_fraction": progress,
		"planner_to_refiner_norm": planner_norm,
		"refiner_to_solver_norm": refiner_norm,
		"feedback_to_planner_norm": feedback_norm,
		"planner_norm_delta": planner_norm - previous["planner_to_refiner_norm"],
		"refiner_norm_delta": refiner_norm - previous["refiner_to_solver_norm"],
		"feedback_norm_delta": feedback_norm - previous["feedback_to_planner_norm"],
		"latent_drift": abs(refiner_norm - planner_norm),
		"feedback_cosine_to_previous": cosine,
		"is_first_round": float(round_index == 0),
		"is_last_round": float(is_last),
		"label": label,
	}
	return record


def generate_records(entry_count: int, seed: int) -> List[Dict[str, float | int | str | bool]]:
	if entry_count <= 0:
		raise ValueError("--entries must be positive")
	entry_count = min(entry_count, MAX_ENTRIES)
	rng = random.Random(seed)
	records: List[Dict[str, float | int | str | bool]] = []
	sample_index = 0

	while len(records) < entry_count:
		max_depth = rng.randint(2, 5)
		previous = {
			"planner_to_refiner_norm": rng.uniform(0.4, 1.6),
			"refiner_to_solver_norm": rng.uniform(0.4, 1.6),
			"feedback_to_planner_norm": rng.uniform(0.4, 1.6),
			"feedback_cosine_to_previous": rng.uniform(-0.2, 0.9),
		}
		for round_index in range(max_depth):
			if len(records) >= entry_count:
				break
			record = _round_record(rng, sample_index, round_index, max_depth, previous)
			records.append(record)
			previous = {
				"planner_to_refiner_norm": float(record["planner_to_refiner_norm"]),
				"refiner_to_solver_norm": float(record["refiner_to_solver_norm"]),
				"feedback_to_planner_norm": float(record["feedback_to_planner_norm"]),
				"feedback_cosine_to_previous": float(record["feedback_cosine_to_previous"]),
			}
		sample_index += 1

	return records


def validate_records(records: List[Dict[str, float | int | str | bool]]) -> None:
	if not records or len(records) > MAX_ENTRIES:
		raise ValueError(f"Expected between 1 and {MAX_ENTRIES} records")
	required = set(FEATURE_NAMES) | {"label", "sample_id", "round_idx", "max_depth"}
	for index, record in enumerate(records, start=1):
		missing = required.difference(record)
		if missing:
			raise ValueError(f"Record {index} is missing fields: {sorted(missing)}")
		if record["label"] not in (0, 1):
			raise ValueError(f"Record {index} has a non-binary label")
	if {int(record["label"]) for record in records} != {0, 1}:
		raise ValueError("Smoke dataset must contain both label classes")


def write_jsonl(records: List[Dict[str, float | int | str | bool]], output: str) -> Path:
	output_path = Path(output)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as handle:
		for record in records:
			handle.write(json.dumps(record, separators=(",", ":")) + "\n")
	return output_path


def main(argv=None) -> None:
	args = parse_args(argv)
	records = generate_records(args.entries, args.seed)
	validate_records(records)
	output_path = write_jsonl(records, args.output)
	labels = {label: sum(int(record["label"]) == label for record in records) for label in (0, 1)}
	print(f"wrote={output_path} entries={len(records)} labels={labels}")


if __name__ == "__main__":
	main()
