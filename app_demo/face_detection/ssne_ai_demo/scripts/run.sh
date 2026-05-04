#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
APP_DIR="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
APP_BIN="$APP_DIR/ssne_ai_demo"

cd "$APP_DIR"

if [ ! -f "$APP_BIN" ]; then
    echo "[ERROR] Binary not found: $APP_BIN"
    exit 1
fi

if [ -f "/lib/modules/$(uname -r)/extra/uart_kmod.ko" ]; then
    insmod "/lib/modules/$(uname -r)/extra/uart_kmod.ko" 2>/dev/null || true
fi

chmod +x "$APP_BIN"
exec "$APP_BIN"
