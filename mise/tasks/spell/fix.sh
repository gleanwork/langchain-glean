#!/usr/bin/env bash
#MISE description="Fix spelling"
set -euo pipefail

uv run codespell --toml pyproject.toml -w --skip=docs/
