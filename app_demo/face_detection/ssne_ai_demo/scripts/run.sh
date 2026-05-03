#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR/.."

if [ ! -f "./ssne_ai_demo" ]; then
    echo "ssne_ai_demo not found in: $PWD" >&2
    exit 1
fi

chmod +x ./ssne_ai_demo
exec ./ssne_ai_demo
