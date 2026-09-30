const camera = document.querySelector("#camera");
const resultCanvas = document.querySelector("#result");
const overlayCanvas = document.querySelector("#overlay");
const empty = document.querySelector("#empty");
const cameraButton = document.querySelector("#camera-button");
const captureButton = document.querySelector("#capture-button");
const liveButton = document.querySelector("#live-button");
const fileInput = document.querySelector("#file-input");
const statusLabel = document.querySelector("#status");
const errorLabel = document.querySelector("#error");
const threshold = document.querySelector("#threshold");
const thresholdValue = document.querySelector("#threshold-value");

const LIVE_FRAME_INTERVAL_MS = 200;
const LIVE_MAX_WIDTH = 960;
let stream = null;
let liveMode = false;
let requestInFlight = false;
let liveFrameHandle = null;
let lastLiveCaptureAt = 0;
let previousLiveResultAt = null;
let smoothedFps = null;

threshold.addEventListener("input", () => {
  thresholdValue.textContent = `${threshold.value}%`;
});

function showError(message) {
  errorLabel.textContent = message;
  errorLabel.hidden = !message;
}

function setStatus(message, busy = false) {
  statusLabel.textContent = message;
  statusLabel.classList.toggle("busy", busy);
}

function updateControls() {
  captureButton.disabled = !stream || requestInFlight || liveMode;
  liveButton.disabled = !stream || (!liveMode && requestInFlight);
  liveButton.textContent = liveMode ? "Stop live" : "Start live";
  liveButton.classList.toggle("active", liveMode);
  fileInput.disabled = requestInFlight || liveMode;
  cameraButton.disabled = requestInFlight || liveMode;
}

function resetMetrics() {
  document.querySelector("#total").textContent = "—";
  document.querySelector("#live-count").textContent = "—";
  document.querySelector("#dead-count").textContent = "—";
  document.querySelector("#latency").textContent = "—";
  document.querySelector("#fps").textContent = "—";
  document.querySelector("#device").textContent = "—";
}

function clearOverlay() {
  const context = overlayCanvas.getContext("2d");
  context.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  overlayCanvas.hidden = true;
}

async function startCamera() {
  showError("");
  stopLiveInspection(true);
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = null;
  updateControls();

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });
    camera.srcObject = stream;
    await camera.play();
    stream.getVideoTracks()[0]?.addEventListener("ended", () => {
      stopLiveInspection();
      stream = null;
      updateControls();
      setStatus("Camera stopped");
    });
    camera.hidden = false;
    resultCanvas.hidden = true;
    clearOverlay();
    empty.hidden = true;
    resetMetrics();
    cameraButton.textContent = "Restart camera";
    setStatus("Camera active");
  } catch (error) {
    stream = null;
    showError("Camera access failed. Allow camera permission or select a photo instead.");
    setStatus("Camera unavailable");
  } finally {
    updateControls();
  }
}

cameraButton.addEventListener("click", startCamera);

captureButton.addEventListener("click", async () => {
  try {
    const blob = await captureCameraFrame(1280, 0.9);
    await inspectSingle(blob);
  } catch (error) {
    showError(error.message || "Could not capture the camera frame.");
  }
});

liveButton.addEventListener("click", () => {
  if (liveMode) stopLiveInspection();
  else startLiveInspection();
});

fileInput.addEventListener("change", async () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  stopLiveInspection(true);
  await inspectSingle(file);
  fileInput.value = "";
});

function captureCameraFrame(maxWidth, quality) {
  if (!stream || camera.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    return Promise.reject(new Error("The camera is not ready yet."));
  }
  const scale = Math.min(1, maxWidth / camera.videoWidth);
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(camera.videoWidth * scale));
  canvas.height = Math.max(1, Math.round(camera.videoHeight * scale));
  canvas.getContext("2d").drawImage(camera, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error("Frame encoding failed."))),
      "image/jpeg",
      quality,
    );
  });
}

async function requestEvaluation(blob) {
  const response = await fetch(
    `/api/evaluate?threshold=${Number(threshold.value) / 100}`,
    {
      method: "POST",
      headers: { "Content-Type": blob.type || "image/jpeg" },
      body: blob,
    },
  );
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || "Inspection failed.");
  return payload;
}

async function inspectSingle(blob) {
  if (!blob || requestInFlight) return;
  requestInFlight = true;
  updateControls();
  setStatus("Inspecting…", true);
  showError("");
  try {
    const payload = await requestEvaluation(blob);
    await drawSnapshot(blob, payload);
    updateMetrics(payload, null);
    setStatus("Inspection complete");
  } catch (error) {
    showError(error.message || "Inspection failed.");
    setStatus("Inspection failed");
  } finally {
    requestInFlight = false;
    updateControls();
  }
}

function startLiveInspection() {
  if (!stream || liveMode) return;
  liveMode = true;
  previousLiveResultAt = null;
  smoothedFps = null;
  lastLiveCaptureAt = 0;
  camera.hidden = false;
  resultCanvas.hidden = true;
  clearOverlay();
  overlayCanvas.hidden = false;
  showError("");
  setStatus("Starting live…", true);
  updateControls();
  liveFrameHandle = requestAnimationFrame(liveLoop);
}

function stopLiveInspection(preserveStatus = false) {
  liveMode = false;
  if (liveFrameHandle !== null) cancelAnimationFrame(liveFrameHandle);
  liveFrameHandle = null;
  previousLiveResultAt = null;
  smoothedFps = null;
  clearOverlay();
  document.querySelector("#fps").textContent = "—";
  if (!preserveStatus && stream) setStatus("Camera active");
  updateControls();
}

function liveLoop(timestamp) {
  if (!liveMode) return;
  if (!requestInFlight && timestamp - lastLiveCaptureAt >= LIVE_FRAME_INTERVAL_MS) {
    lastLiveCaptureAt = timestamp;
    void inspectLiveFrame();
  }
  liveFrameHandle = requestAnimationFrame(liveLoop);
}

async function inspectLiveFrame() {
  requestInFlight = true;
  updateControls();
  try {
    const blob = await captureCameraFrame(LIVE_MAX_WIDTH, 0.75);
    const payload = await requestEvaluation(blob);
    if (!liveMode) return;

    drawLiveOverlay(payload);
    const now = performance.now();
    if (previousLiveResultAt !== null) {
      const instantFps = 1000 / (now - previousLiveResultAt);
      smoothedFps = smoothedFps === null
        ? instantFps
        : smoothedFps * 0.7 + instantFps * 0.3;
    }
    previousLiveResultAt = now;
    updateMetrics(payload, smoothedFps);
    setStatus(smoothedFps ? `Live · ${smoothedFps.toFixed(1)} FPS` : "Live inspection");
  } catch (error) {
    showError(error.message || "Live inspection failed.");
    stopLiveInspection(true);
    setStatus("Live inspection stopped");
  } finally {
    requestInFlight = false;
    updateControls();
  }
}

function drawDetections(context, payload) {
  const scale = Math.max(1, payload.image.width / 1000);
  for (const detection of payload.detections) {
    const [x1, y1, x2, y2] = detection.box;
    const color = detection.class_id === 1 ? "#58d68d" : "#ff7b72";
    const label = `${detection.label} ${Math.round(detection.confidence * 100)}%`;
    context.strokeStyle = color;
    context.lineWidth = 3 * scale;
    context.strokeRect(x1, y1, x2 - x1, y2 - y1);
    context.font = `700 ${18 * scale}px sans-serif`;
    const textWidth = context.measureText(label).width;
    const labelHeight = 27 * scale;
    context.fillStyle = color;
    context.fillRect(x1, Math.max(0, y1 - labelHeight), textWidth + 12 * scale, labelHeight);
    context.fillStyle = "#101b15";
    context.fillText(label, x1 + 6 * scale, Math.max(19 * scale, y1 - 7 * scale));
  }
}

async function drawSnapshot(blob, payload) {
  const bitmap = await createImageBitmap(blob);
  resultCanvas.width = bitmap.width;
  resultCanvas.height = bitmap.height;
  const context = resultCanvas.getContext("2d");
  context.drawImage(bitmap, 0, 0);
  drawDetections(context, payload);
  camera.hidden = true;
  clearOverlay();
  empty.hidden = true;
  resultCanvas.hidden = false;
  bitmap.close();
}

function drawLiveOverlay(payload) {
  overlayCanvas.width = payload.image.width;
  overlayCanvas.height = payload.image.height;
  const context = overlayCanvas.getContext("2d");
  context.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  drawDetections(context, payload);
  overlayCanvas.hidden = false;
}

function updateMetrics(payload, fps) {
  document.querySelector("#total").textContent = payload.total;
  document.querySelector("#live-count").textContent = payload.counts["live knot"];
  document.querySelector("#dead-count").textContent = payload.counts["dead knot"];
  document.querySelector("#latency").textContent = `${payload.latency_ms} ms`;
  document.querySelector("#fps").textContent = fps ? `${fps.toFixed(1)} FPS` : "—";
  document.querySelector("#device").textContent = payload.device.toUpperCase();
}

document.addEventListener("visibilitychange", () => {
  if (document.hidden && liveMode) stopLiveInspection();
});

updateControls();
