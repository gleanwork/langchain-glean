#!/usr/bin/env bash
#MISE description="Run linters on changed files"
set -euo pipefail

PYTHON_FILES=$(git diff --relative --name-only --diff-filter=d master | grep -E '\.py$|\.ipynb$' || true)
if [ -n "$PYTHON_FILES" ]; then
  uv run ruff check $PYTHON_FILES
  uv run ruff format $PYTHON_FILES --diff
  mkdir -p .mypy_cache && uv run mypy $PYTHON_FILES --cache-dir .mypy_cache
fi
