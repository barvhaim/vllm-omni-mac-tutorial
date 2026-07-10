#!/usr/bin/env bash
set -euo pipefail
command -v python3 >/dev/null || { echo "Install Python 3.11 or 3.12 first."; exit 1; }
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
echo "Ready. Run: source .venv/bin/activate && jupyter lab"
