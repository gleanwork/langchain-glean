#!/usr/bin/env bash
#MISE description="Run ruff linter"
#MISE hide=true
set -euo pipefail

uv run ruff check . --exclude docs/ --exclude tests/
