"""Single-image inference for the trained TimberSight detector."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from timbersight.modeling.model import build_detector
from timbersight.modeling.train import select_device
from timbersight.modeling.transforms import detection_transforms

LABELS = {1: "live knot", 2: "dead knot"}


def resolve_checkpoint(explicit: str | Path | None = None) -> Path:
    """Resolve an explicit, configured, or most-recent local checkpoint."""
    configured = explicit or os.environ.get("TIMBERSIGHT_CHECKPOINT")
    if configured:
        checkpoint = Path(configured).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint

    candidates = list(Path("artifacts/runs").glob("*/model.pt"))
    if not candidates:
        raise FileNotFoundError(
            "No checkpoint found. Set TIMBERSIGHT_CHECKPOINT to a model.pt file."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def format_detections(
    prediction: dict[str, torch.Tensor],
    threshold: float,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    """Convert torchvision output to JSON-safe, image-bounded detections."""
    detections: list[dict[str, Any]] = []
    for box, label, score in zip(
        prediction["boxes"], prediction["labels"], prediction["scores"]
    ):
        confidence = float(score.item())
        class_id = int(label.item())
        if confidence < threshold or class_id not in LABELS:
            continue
        x1, y1, x2, y2 = (float(value) for value in box.tolist())
        detections.append(
            {
                "class_id": class_id,
                "label": LABELS[class_id],
                "confidence": round(confidence, 4),
                "box": [
                    round(max(0.0, min(x1, width)), 2),
                    round(max(0.0, min(y1, height)), 2),
                    round(max(0.0, min(x2, width)), 2),
                    round(max(0.0, min(y2, height)), 2),
                ],
            }
        )
    return detections


class TimberSightDetector:
    """Thread-safe lazy loader for photo inspection requests."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        device: str = "auto",
    ) -> None:
        self.checkpoint = resolve_checkpoint(checkpoint)
        self.device = select_device(device)
        self._model = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._transform = detection_transforms(training=False)

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            model = build_detector(pretrained=False)
            state_dict = torch.load(
                self.checkpoint, map_location=self.device, weights_only=True
            )
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self._model = model

    def predict(self, image: Image.Image, threshold: float = 0.5) -> dict[str, Any]:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.load()
        rgb_image = image.convert("RGB")
        tensor = self._transform(pil_to_tensor(rgb_image)).to(self.device)

        started = time.perf_counter()
        with self._inference_lock, torch.inference_mode():
            prediction = self._model([tensor])[0]
        latency_ms = (time.perf_counter() - started) * 1000

        detections = format_detections(
            prediction, threshold, rgb_image.width, rgb_image.height
        )
        counts = {
            "live knot": sum(item["class_id"] == 1 for item in detections),
            "dead knot": sum(item["class_id"] == 2 for item in detections),
        }
        return {
            "image": {"width": rgb_image.width, "height": rgb_image.height},
            "detections": detections,
            "counts": counts,
            "total": len(detections),
            "latency_ms": round(latency_ms, 1),
            "device": str(self.device),
        }
