#!/usr/bin/env bash

set -e
PROJECT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec "$PROJECT_DIR/timbersight.sh" stop

