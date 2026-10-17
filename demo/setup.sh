#!/usr/bin/env bash
# One-time setup: creates .venv here and installs the demo's packages (CPU is enough).
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Setup done. Test with:  source .venv/bin/activate && python app.py --mock"
