#!/usr/bin/env bash
#MISE description="Run lint autofixers on test files"
set -euo pipefail

uv run ruff check tests --fix
uv run ruff format tests
uv run ruff check --select I --fix tests
