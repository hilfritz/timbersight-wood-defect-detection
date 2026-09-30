const camera = document.querySelector("#camera");
const resultCanvas = document.querySelector("#result");
const empty = document.querySelector("#empty");
const cameraButton = document.querySelector("#camera-button");
const captureButton = document.querySelector("#capture-button");
const fileInput = document.querySelector("#file-input");
const statusLabel = document.querySelector("#status");
const errorLabel = document.querySelector("#error");
const threshold = document.querySelector("#threshold");
const thresholdValue = document.querySelector("#threshold-value");
let stream = null;

threshold.addEventListener("input", () => {
  thresholdValue.textContent = `${threshold.value}%`;
});

function setBusy(isBusy, message = "Ready") {
  statusLabel.textContent = message;
  statusLabel.classList.toggle("busy", isBusy);
  captureButton.disabled = isBusy || !stream;
  fileInput.disabled = isBusy;
  cameraButton.disabled = isBusy;
}

function showError(message) {
  errorLabel.textContent = message;
  errorLabel.hidden = !message;
}

cameraButton.addEventListener("click", async () => {
  showError("");
  try {
    if (stream) stream.getTracks().forEach((track) => track.stop());
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" } },
      audio: false,
    });
    camera.srcObject = stream;
    camera.hidden = false;
    resultCanvas.hidden = true;
    empty.hidden = true;
    captureButton.disabled = false;
    cameraButton.textContent = "Restart camera";
    statusLabel.textContent = "Camera active";
  } catch (error) {
    showError("Camera access failed. Allow camera permission or select a photo instead.");
  }
});

captureButton.addEventListener("click", async () => {
  const canvas = document.createElement("canvas");
  canvas.width = camera.videoWidth;
  canvas.height = camera.videoHeight;
  canvas.getContext("2d").drawImage(camera, 0, 0);
  canvas.toBlob((blob) => inspect(blob), "image/jpeg", 0.9);
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (file) inspect(file);
});

async function inspect(blob) {
  if (!blob) return;
  setBusy(true, "Inspecting…");
  showError("");
  try {
    const response = await fetch(`/api/evaluate?threshold=${Number(threshold.value) / 100}`, {
      method: "POST",
      headers: { "Content-Type": blob.type || "image/jpeg" },
      body: blob,
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Inspection failed.");
    await drawResult(blob, payload);
    updateMetrics(payload);
    statusLabel.textContent = "Inspection complete";
  } catch (error) {
    showError(error.message || "Inspection failed.");
    statusLabel.textContent = "Inspection failed";
  } finally {
    setBusy(false, statusLabel.textContent);
  }
}

async function drawResult(blob, payload) {
  const bitmap = await createImageBitmap(blob);
  resultCanvas.width = bitmap.width;
  resultCanvas.height = bitmap.height;
  const context = resultCanvas.getContext("2d");
  context.drawImage(bitmap, 0, 0);
  const scale = Math.max(1, bitmap.width / 1000);

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

  camera.hidden = true;
  empty.hidden = true;
  resultCanvas.hidden = false;
  bitmap.close();
}

function updateMetrics(payload) {
  document.querySelector("#total").textContent = payload.total;
  document.querySelector("#live-count").textContent = payload.counts["live knot"];
  document.querySelector("#dead-count").textContent = payload.counts["dead knot"];
  document.querySelector("#latency").textContent = `${payload.latency_ms} ms`;
  document.querySelector("#device").textContent = payload.device.toUpperCase();
}

