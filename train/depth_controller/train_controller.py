from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch import nn

try:
	from .controller_helper import FEATURE_NAMES
	from .depth_controller import DepthController
except ImportError:
	from controller_helper import FEATURE_NAMES
	from depth_controller import DepthController


def parse_args(argv=None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train the sequential depth controller from JSONL traces.")
	parser.add_argument("--data", required=True, help="Path to controller_dataset.jsonl")
	parser.add_argument("--output", required=True, help="Path for the trained controller checkpoint")
	parser.add_argument("--epochs", type=int, default=30)
	parser.add_argument("--batch-size", type=int, default=256)
	parser.add_argument("--learning-rate", type=float, default=1e-3)
	parser.add_argument("--weight-decay", type=float, default=1e-4)
	parser.add_argument("--validation-fraction", type=float, default=0.2)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--device", default=None)
	return parser.parse_args(argv)


def _number(value: Any) -> float:
	try:
		return float(value)
	except (TypeError, ValueError) as exc:
		raise ValueError(f"Expected numeric feature value, got {value!r}") from exc


def load_records(path: str) -> List[Dict[str, Any]]:
	records: List[Dict[str, Any]] = []
	with Path(path).open("r", encoding="utf-8") as handle:
		for line_number, line in enumerate(handle, start=1):
			if not line.strip():
				continue
			try:
				record = json.loads(line)
			except json.JSONDecodeError as exc:
				raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
			missing = [name for name in FEATURE_NAMES + ["label"] if name not in record]
			if missing:
				raise ValueError(f"Line {line_number} is missing fields: {missing}")
			records.append(record)
	if not records:
		raise ValueError(f"No records found in {path}")
	return records


def split_by_sample(
	records: Sequence[Dict[str, Any]], validation_fraction: float, seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
	if not 0.0 < validation_fraction < 1.0:
		raise ValueError("--validation-fraction must be between 0 and 1")
	sample_ids = list(dict.fromkeys(str(record.get("sample_id", index)) for index, record in enumerate(records)))
	random.Random(seed).shuffle(sample_ids)
	validation_count = max(1, round(len(sample_ids) * validation_fraction))
	validation_ids = set(sample_ids[:validation_count])
	validation = [record for record in records if str(record.get("sample_id")) in validation_ids]
	training = [record for record in records if str(record.get("sample_id")) not in validation_ids]
	if not training or not validation:
		raise ValueError("The dataset needs at least two distinct sample_id groups")
	return training, validation


def tensorize(records: Sequence[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
	features = torch.tensor(
		[[_number(record[name]) for name in FEATURE_NAMES] for record in records],
		dtype=torch.float32,
	)
	labels = torch.tensor([_number(record["label"]) for record in records], dtype=torch.float32)
	if not torch.all((labels == 0) | (labels == 1)):
		raise ValueError("Controller labels must be binary 0/1 values")
	return features, labels


def evaluate(
	model: nn.Module,
	features: torch.Tensor,
	labels: torch.Tensor,
	criterion: nn.Module,
) -> Tuple[float, float]:
	model.eval()
	with torch.no_grad():
		logits = model(features).squeeze(-1)
		loss = criterion(logits, labels).item()
		accuracy = ((torch.sigmoid(logits) >= 0.5) == labels.bool()).float().mean().item()
	return loss, accuracy


def train(args: argparse.Namespace) -> None:
	if args.epochs <= 0 or args.batch_size <= 0:
		raise ValueError("--epochs and --batch-size must be positive")
	torch.manual_seed(args.seed)
	records = load_records(args.data)
	training_records, validation_records = split_by_sample(records, args.validation_fraction, args.seed)
	train_features, train_labels = tensorize(training_records)
	validation_features, validation_labels = tensorize(validation_records)

	feature_mean = train_features.mean(dim=0)
	feature_std = train_features.std(dim=0, unbiased=False).clamp_min(1e-6)
	train_features = (train_features - feature_mean) / feature_std
	validation_features = (validation_features - feature_mean) / feature_std

	device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
	model = DepthController(hidden_size=len(FEATURE_NAMES), adapter_type="sequential").to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
	positive_count = train_labels.sum().item()
	negative_count = len(train_labels) - positive_count
	pos_weight = torch.tensor([negative_count / max(positive_count, 1.0)], device=device)
	criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

	train_features = train_features.to(device)
	train_labels = train_labels.to(device)
	validation_features = validation_features.to(device)
	validation_labels = validation_labels.to(device)

	for epoch in range(1, args.epochs + 1):
		model.train()
		permutation = torch.randperm(len(train_features), device=device)
		total_loss = 0.0
		for start in range(0, len(train_features), args.batch_size):
			indices = permutation[start : start + args.batch_size]
			optimizer.zero_grad(set_to_none=True)
			logits = model(train_features[indices]).squeeze(-1)
			loss = criterion(logits, train_labels[indices])
			loss.backward()
			optimizer.step()
			total_loss += loss.item() * len(indices)

		validation_loss, validation_accuracy = evaluate(model, validation_features, validation_labels, criterion)
		print(
			f"epoch={epoch:03d} train_loss={total_loss / len(train_features):.4f} "
			f"val_loss={validation_loss:.4f} val_accuracy={validation_accuracy:.4f}",
			flush=True,
		)

	output_path = Path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(
		{
			"model_state_dict": model.state_dict(),
			"hidden_size": len(FEATURE_NAMES),
			"adapter_type": "sequential",
			"feature_names": FEATURE_NAMES,
			"feature_mean": feature_mean.cpu(),
			"feature_std": feature_std.cpu(),
			"validation_loss": validation_loss,
			"validation_accuracy": validation_accuracy,
		},
		output_path,
	)
	print(f"saved={output_path}", flush=True)


if __name__ == "__main__":
	train(parse_args())
