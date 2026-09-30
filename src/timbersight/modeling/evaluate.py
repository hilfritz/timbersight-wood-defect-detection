"""Evaluate a completed TimberSight checkpoint on a named dataset split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from timbersight.data.validate import load_config
from timbersight.modeling.dataset import (
    VNWoodKnotDetection,
    collate_detection_batch,
)
from timbersight.modeling.model import build_detector
from timbersight.modeling.train import evaluate, select_device
from timbersight.modeling.transforms import detection_transforms


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path, default=Path("config/dataset.toml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --overwrite")

    device = select_device(args.device)
    config = load_config(args.dataset_config.resolve())
    dataset = VNWoodKnotDetection(
        config, args.split, detection_transforms(training=False)
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_detection_batch,
    )

    model = build_detector(pretrained=False)
    state_dict = torch.load(
        args.checkpoint, map_location=device, weights_only=True
    )
    model.load_state_dict(state_dict)
    model.to(device)
    metrics = evaluate(model, loader, device, args.score_threshold)

    with args.checkpoint.open("rb") as stream:
        checkpoint_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    result = {
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "device": str(device),
        **metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

