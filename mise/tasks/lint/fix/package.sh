#!/usr/bin/env bash
#MISE description="Run lint autofixers on package files"
set -euo pipefail

uv run ruff check langchain_glean --fix
uv run ruff format langchain_glean
uv run ruff check --select I --fix langchain_glean
