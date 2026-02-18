#!/usr/bin/env bash
#MISE description="Run ruff autofixer"
#MISE hide=true
set -euo pipefail

uv run ruff check . --fix --exclude docs/ --exclude tests/
