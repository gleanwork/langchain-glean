#!/usr/bin/env bash
#MISE description="Run linters on test files"
set -euo pipefail

uv run ruff check tests
uv run ruff format tests --diff
mkdir -p .mypy_cache_test && uv run mypy tests --cache-dir .mypy_cache_test
