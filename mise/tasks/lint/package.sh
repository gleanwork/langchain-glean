#!/usr/bin/env bash
#MISE description="Run linters on package files"
set -euo pipefail

uv run ruff check langchain_glean
uv run ruff format langchain_glean --diff
mkdir -p .mypy_cache && uv run mypy langchain_glean --cache-dir .mypy_cache
