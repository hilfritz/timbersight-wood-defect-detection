# TimberSight camera inspection

The TimberSight web application captures a camera frame or accepts an uploaded
wood image, runs an existing trained detector and overlays live-knot and
dead-knot predictions. Continuous mode repeats this process while keeping only
one inference request in flight. The application performs inference only; it
does not retrain the model.

## Requirements

- Python 3.11 or newer
- A trained TimberSight checkpoint
- A browser with camera permission for live capture

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[ml]" -r requirements-web.txt
```

## Recommended start and stop commands

From the project directory:

```bash
export TIMBERSIGHT_CHECKPOINT="artifacts/runs/RUN_ID/model.pt"
./script_timbersight_start.sh
```

The start helper waits for the health endpoint and then opens the website. It
uses port `8001` by default. To select another port:

```bash
TIMBERSIGHT_PORT=8000 ./script_timbersight_start.sh
```

Manage the local service with:

```bash
./script_timbersight_check.sh
./script_timbersight_open.sh
./script_timbersight_stop.sh
```

The unified helper also supports `start`, `stop`, `status`, `restart`, `open`
and `logs`:

```bash
./timbersight.sh status
./timbersight.sh logs
```

## Manual start

```bash
TIMBERSIGHT_CHECKPOINT="artifacts/runs/RUN_ID/model.pt" \
  python -m uvicorn timbersight.web.app:app \
  --app-dir src \
  --host 127.0.0.1 \
  --port 8000
```

Open <http://127.0.0.1:8000> and allow camera permission, or select a saved
image. Choose **Capture & inspect** for one frame or **Start live** for
continuous inspection.

The page resizes live frames to a maximum width of 960 pixels. The results panel
reports class counts, smoothed live frame rate, model latency and the selected
compute device.

## Processor selection

`TIMBERSIGHT_DEVICE` accepts `cpu`, `cuda`, `mps` or `auto`. The default is
`auto`.

```bash
TIMBERSIGHT_DEVICE=mps ./script_timbersight_start.sh
```

## API

### Health check

```text
GET /api/health
```

### Evaluate an image

```text
POST /api/evaluate?threshold=0.5
Content-Type: image/jpeg
```

The request body contains raw JPEG, PNG or WebP bytes. The response contains
image dimensions, class counts, bounding boxes, confidence scores, inference
latency and the selected compute device.

Example:

```bash
curl --request POST \
  --header "Content-Type: image/jpeg" \
  --data-binary @wood.jpg \
  "http://127.0.0.1:8000/api/evaluate?threshold=0.5"
```

## Deployment note

Camera access normally requires `localhost` or HTTPS. The current application is
designed as a local demonstration application. Public or multi-device deployment
would additionally require HTTPS, authentication, request-size limits,
observability and validation on representative production data.
