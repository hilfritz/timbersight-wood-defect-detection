#!/usr/bin/env bash

set -u

PROJECT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PYTHON="$PROJECT_DIR/.venv/bin/python"
CHECKPOINT="${TIMBERSIGHT_CHECKPOINT:-$PROJECT_DIR/artifacts/runs/20260930T035750Z/model.pt}"
HOST="${TIMBERSIGHT_HOST:-127.0.0.1}"
PORT="${TIMBERSIGHT_PORT:-8001}"
URL="http://$HOST:$PORT"
RUNTIME_DIR="$PROJECT_DIR/.runtime"
PID_FILE="$RUNTIME_DIR/timbersight.pid"
LOG_FILE="$RUNTIME_DIR/timbersight.log"

usage() {
  cat <<'EOF'
Usage: ./timbersight.sh COMMAND

Commands:
  start    Start TimberSight and open it in the default browser
  stop     Stop TimberSight
  status   Check whether TimberSight is running
  check    Alias for status
  restart  Stop, start, and open TimberSight
  open     Open the website without starting the server
  logs     Show the latest server log lines
EOF
}

read_pid() {
  if [[ -f "$PID_FILE" ]]; then
    tr -dc '0-9' < "$PID_FILE"
  fi
}

is_running() {
  local pid
  pid="$(read_pid)"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

open_website() {
  if [[ "${TIMBERSIGHT_NO_OPEN:-0}" == "1" ]]; then
    return
  fi

  if command -v open >/dev/null 2>&1; then
    open "$URL"
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$URL" >/dev/null 2>&1 &
  elif command -v cmd.exe >/dev/null 2>&1; then
    cmd.exe /c start "" "$URL" >/dev/null 2>&1
  else
    echo "Open $URL in your browser."
  fi
}

status_server() {
  if is_running; then
    local pid
    pid="$(read_pid)"
    if curl --fail --silent --max-time 2 "$URL/api/health" >/dev/null 2>&1; then
      echo "TimberSight is running (PID $pid): $URL"
      return 0
    fi
    echo "TimberSight process $pid is starting, but the API is not ready yet."
    return 1
  fi

  echo "TimberSight is stopped."
  return 1
}

start_server() {
  if is_running; then
    echo "TimberSight is already running: $URL"
    open_website
    return 0
  fi

  if [[ ! -x "$PYTHON" ]]; then
    echo "Python environment not found: $PYTHON" >&2
    echo "Create the project virtual environment before starting TimberSight." >&2
    return 1
  fi
  if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Model checkpoint not found: $CHECKPOINT" >&2
    return 1
  fi

  mkdir -p "$RUNTIME_DIR"
  echo "Starting TimberSight…"
  nohup env \
    TIMBERSIGHT_CHECKPOINT="$CHECKPOINT" \
    TIMBERSIGHT_DEVICE="${TIMBERSIGHT_DEVICE:-auto}" \
    "$PYTHON" -m uvicorn timbersight.web.app:app \
      --app-dir "$PROJECT_DIR/src" \
      --host "$HOST" \
      --port "$PORT" \
      >>"$LOG_FILE" 2>&1 &
  printf '%s\n' "$!" > "$PID_FILE"

  local attempt
  for attempt in {1..40}; do
    if curl --fail --silent --max-time 2 "$URL/api/health" >/dev/null 2>&1; then
      echo "TimberSight is ready: $URL"
      open_website
      return 0
    fi
    if ! is_running; then
      echo "TimberSight failed to start. Check: $LOG_FILE" >&2
      return 1
    fi
    sleep 0.25
  done

  echo "TimberSight is still starting. Check with: ./timbersight.sh status" >&2
  echo "Logs: $LOG_FILE" >&2
  return 1
}

stop_server() {
  if ! is_running; then
    echo "TimberSight is already stopped."
    return 0
  fi

  local pid attempt
  pid="$(read_pid)"
  kill "$pid"
  for attempt in {1..40}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "TimberSight stopped."
      return 0
    fi
    sleep 0.25
  done

  echo "TimberSight did not stop within 10 seconds (PID $pid)." >&2
  return 1
}

show_logs() {
  if [[ -f "$LOG_FILE" ]]; then
    tail -n 40 "$LOG_FILE"
  else
    echo "No server log exists yet."
  fi
}

case "${1:-}" in
  start)
    start_server
    ;;
  stop)
    stop_server
    ;;
  status|check)
    status_server
    ;;
  restart)
    stop_server && start_server
    ;;
  open)
    open_website
    ;;
  logs)
    show_logs
    ;;
  help|-h|--help|"")
    usage
    ;;
  *)
    echo "Unknown command: $1" >&2
    usage >&2
    exit 2
    ;;
esac

