#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_LIB="$ROOT/.venv/lib"

# WSLg provides versioned EGL/GL libraries, while ModernGL requests unversioned names.
ln -sfn /usr/lib/x86_64-linux-gnu/libEGL.so.1 "$VENV_LIB/libEGL.so"
ln -sfn /usr/lib/x86_64-linux-gnu/libGL.so.1 "$VENV_LIB/libGL.so"
export LD_LIBRARY_PATH="$VENV_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$ROOT"
exec .venv/bin/python -m src.data_collection.keyboard_capture "$@"
