#!/usr/bin/env bash
#MISE description="Check spelling"
set -euo pipefail

uv run codespell --toml pyproject.toml --skip=docs/
