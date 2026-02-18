#!/usr/bin/env bash
#MISE description="Run ruff formatter"
#MISE hide=true
set -euo pipefail

uv run ruff format .
