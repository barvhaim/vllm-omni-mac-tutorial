#!/usr/bin/env bash
set -euo pipefail
command -v uv >/dev/null || { echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }
uv sync
echo "Ready. Run: uv run jupyter lab"
