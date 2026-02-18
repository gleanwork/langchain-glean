#!/usr/bin/env bash
#MISE description="Set up local environment (install dependencies)"
#MISE sources=["pyproject.toml"]
#MISE outputs=[".venv"]
set -euo pipefail

uv sync --all-extras
