#!/usr/bin/env bash
#MISE description="Check code formatting"
#MISE hide=true
set -euo pipefail

uv run ruff format . --diff
