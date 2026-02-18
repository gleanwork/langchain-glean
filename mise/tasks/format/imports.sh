#!/usr/bin/env bash
#MISE description="Sort imports"
#MISE hide=true
set -euo pipefail

uv run ruff check --select I --fix .
