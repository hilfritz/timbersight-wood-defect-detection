"""Train and evaluate the reproducible Faster R-CNN baseline."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from timbersight.data.validate import load_config
from timbersight.modeling.dataset import (
    BalancedDetectionBatchSampler,
    VNWoodKnotDetection,
    collate_detection_batch,
)
from timbersight.modeling.metrics import detection_ap50
from timbersight.modeling.model import build_detector
from timbersight.modeling.transforms import detection_transforms


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_device(target: dict[str, torch.Tensor], device: torch.device):
    return {key: value.to(device) for key, value in target.items()}


def train_epoch(
    model, loader, optimizer, device: torch.device, epoch: int
) -> float:
    model.train()
    if hasattr(loader.batch_sampler, "set_epoch"):
        loader.batch_sampler.set_epoch(epoch)
    losses: list[float] = []
    warmup = None
    if epoch == 0 and len(loader) > 1:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0 / 1000.0,
            total_iters=min(1000, len(loader) - 1),
        )
    for batch_index, (images, targets) in enumerate(loader):
        images = [image.to(device) for image in images]
        targets = [_to_device(target, device) for target in targets]
        loss_map = model(images, targets)
        loss = sum(loss_map.values())
        if not torch.isfinite(loss):
            values = {
                name: float(value.detach().cpu()) for name, value in loss_map.items()
            }
            image_ids = [int(target["image_id"].detach().cpu()) for target in targets]
            raise FloatingPointError(
                f"Non-finite loss at batch {batch_index}, image IDs {image_ids}: "
                f"{values}"
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=float(model.gradient_clip_norm)
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(
                f"Non-finite gradient norm at batch {batch_index}: {gradient_norm}"
            )
        optimizer.step()
        if warmup is not None:
            warmup.step()
        losses.append(float(loss.detach().cpu()))
    return sum(losses) / max(len(losses), 1)


@torch.inference_mode()
def evaluate(model, loader, device: torch.device, score_threshold: float):
    model.eval()
    predictions = []
    targets_for_metrics = []
    latencies_ms: list[float] = []
    for images, targets in loader:
        images_on_device = [image.to(device) for image in images]
        started = time.perf_counter()
        outputs = model(images_on_device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - started) * 1000 / len(images))
        predictions.extend(
            [{key: value.detach().cpu() for key, value in output.items()} for output in outputs]
        )
        targets_for_metrics.extend(
            [{key: value.detach().cpu() for key, value in target.items()} for target in targets]
        )
    metrics = detection_ap50(
        predictions, targets_for_metrics, score_threshold=score_threshold
    )
    metrics["mean_latency_ms_per_image"] = round(
        sum(latencies_ms) / max(len(latencies_ms), 1), 3
    )
    metrics["evaluated_images"] = len(targets_for_metrics)
    return metrics


def _git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _load_train_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _balanced_smoke_indices(dataset: VNWoodKnotDetection) -> list[int]:
    indices: list[int] = []
    for category_folder in dataset.config.image_categories:
        index = next(
            sample_index
            for sample_index, sample in enumerate(dataset.samples)
            if sample.category_folder == category_folder
        )
        indices.append(index)
    return indices


def _stratified_indices(
    dataset: VNWoodKnotDetection, maximum: int | None
) -> list[int] | None:
    if maximum is None or maximum >= len(dataset):
        return None
    groups = {
        category: [
            index
            for index, sample in enumerate(dataset.samples)
            if sample.category_folder == category
        ]
        for category in dataset.config.image_categories
    }
    selected: list[int] = []
    while len(selected) < maximum:
        made_progress = False
        for category in dataset.config.image_categories:
            if groups[category] and len(selected) < maximum:
                selected.append(groups[category].pop(0))
                made_progress = True
        if not made_progress:
            break
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path, default=Path("config/dataset.toml"))
    parser.add_argument("--train-config", type=Path, default=Path("config/train.toml"))
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one untrained CPU batch and a two-image evaluation without downloading weights",
    )
    parser.add_argument("--epochs", type=int, help="Override configured epoch count")
    parser.add_argument("--device", help="Override configured device")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    train_config = _load_train_config(args.train_config.resolve())
    torch_cache = Path(train_config["torch_cache"]).resolve()
    torch_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(torch_cache))
    dataset_config = load_config(args.dataset_config.resolve())
    seed = int(train_config["seed"])
    set_reproducible_seed(seed)

    smoke_test = args.smoke_test
    requested_device = args.device or train_config["device"]
    device = torch.device("cpu") if smoke_test else select_device(requested_device)
    pretrained = False if smoke_test else bool(train_config["pretrained"])
    epochs = 1 if smoke_test else int(args.epochs or train_config["epochs"])

    train_dataset = VNWoodKnotDetection(
        dataset_config, "train", detection_transforms(training=True)
    )
    validation_dataset = VNWoodKnotDetection(
        dataset_config, "validation", detection_transforms(training=False)
    )
    if smoke_test:
        train_dataset = Subset(train_dataset, _balanced_smoke_indices(train_dataset))
        validation_dataset = Subset(
            validation_dataset, _balanced_smoke_indices(validation_dataset)
        )
    else:
        train_indices = _stratified_indices(train_dataset, args.max_train_samples)
        validation_indices = _stratified_indices(
            validation_dataset, args.max_validation_samples
        )
        if train_indices is not None:
            train_dataset = Subset(train_dataset, train_indices)
        if validation_indices is not None:
            validation_dataset = Subset(validation_dataset, validation_indices)

    if smoke_test:
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_detection_batch,
        )
    else:
        batch_sampler = BalancedDetectionBatchSampler(
            train_dataset,
            batch_size=int(train_config["batch_size"]),
            seed=seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=int(train_config["num_workers"]),
            collate_fn=collate_detection_batch,
        )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1 if smoke_test else int(train_config["batch_size"]),
        shuffle=False,
        num_workers=0 if smoke_test else int(train_config["num_workers"]),
        collate_fn=collate_detection_batch,
    )

    model = build_detector(pretrained=pretrained).to(device)
    model.gradient_clip_norm = float(train_config["gradient_clip_norm"])
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if train_config["optimizer"] != "sgd":
        raise ValueError(f"Unsupported optimizer: {train_config['optimizer']}")
    optimizer = torch.optim.SGD(
        parameters,
        lr=float(train_config["learning_rate"]),
        momentum=float(train_config["momentum"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(train_config["scheduler_step_size"]),
        gamma=float(train_config["scheduler_gamma"]),
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = "-smoke" if smoke_test else ""
    output_dir = Path(train_config["output_root"]) / f"{run_id}{suffix}"
    output_dir.mkdir(parents=True, exist_ok=False)

    history = []
    for epoch in range(epochs):
        learning_rate = optimizer.param_groups[0]["lr"]
        mean_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        metrics = evaluate(
            model,
            validation_loader,
            device,
            score_threshold=float(train_config["score_threshold"]),
        )
        record = {
            "epoch": epoch + 1,
            "learning_rate": learning_rate,
            "mean_train_loss": mean_loss,
            **metrics,
        }
        history.append(record)
        print(json.dumps(record, indent=2))
        scheduler.step()

    result = {
        "run_id": run_id,
        "smoke_test": smoke_test,
        "seed": seed,
        "device": str(device),
        "pretrained": pretrained,
        "train_images": len(train_dataset),
        "validation_images": len(validation_dataset),
        "git_revision": _git_revision(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "history": history,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    if not smoke_test:
        torch.save(model.state_dict(), output_dir / "model.pt")
    print(f"Run artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
