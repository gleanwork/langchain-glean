#!/usr/bin/env bash
#MISE description="Run mypy type checker"
#MISE hide=true
set -euo pipefail

mkdir -p .mypy_cache && uv run mypy . --cache-dir .mypy_cache --exclude docs/ --exclude tests/
