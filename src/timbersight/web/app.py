"""FastAPI service for camera and uploaded-photo wood inspection."""

from __future__ import annotations

import io
import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from timbersight.inference import TimberSightDetector

STATIC_DIR = Path(__file__).parent / "static"
MAX_IMAGE_BYTES = 15 * 1024 * 1024

app = FastAPI(title="TimberSight Inspection API", version="0.1.0")


@lru_cache(maxsize=1)
def detector() -> TimberSightDetector:
    return TimberSightDetector(
        checkpoint=os.environ.get("TIMBERSIGHT_CHECKPOINT"),
        device=os.environ.get("TIMBERSIGHT_DEVICE", "auto"),
    )


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/styles.css", include_in_schema=False)
def styles() -> FileResponse:
    return FileResponse(STATIC_DIR / "styles.css", media_type="text/css")


@app.get("/app.js", include_in_schema=False)
def javascript() -> FileResponse:
    return FileResponse(STATIC_DIR / "app.js", media_type="text/javascript")


@app.get("/app-realtime.js", include_in_schema=False)
def realtime_javascript() -> FileResponse:
    return FileResponse(STATIC_DIR / "app-realtime.js", media_type="text/javascript")


@app.get("/api/health")
def health() -> dict[str, object]:
    try:
        service = detector()
        return {
            "status": "ready",
            "model_loaded": service.loaded,
            "device": str(service.device),
            "checkpoint": service.checkpoint.name,
        }
    except (FileNotFoundError, RuntimeError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/api/evaluate")
async def evaluate_photo(
    request: Request,
    threshold: float = Query(default=0.5, ge=0.0, le=1.0),
) -> dict[str, object]:
    payload = await request.body()
    if not payload:
        raise HTTPException(status_code=400, detail="No image was provided.")
    if len(payload) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 15 MB limit.")

    try:
        with Image.open(io.BytesIO(payload)) as source:
            image = source.convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=400, detail="Unsupported or invalid image.") from error

    try:
        return detector().predict(image, threshold)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
