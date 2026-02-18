#!/usr/bin/env bash
#MISE description="Run formatters on changed files"
set -euo pipefail

PYTHON_FILES=$(git diff --relative --name-only --diff-filter=d master | grep -E '\.py$|\.ipynb$' || true)
if [ -n "$PYTHON_FILES" ]; then
  uv run ruff format $PYTHON_FILES
  uv run ruff check --select I --fix $PYTHON_FILES
fi
