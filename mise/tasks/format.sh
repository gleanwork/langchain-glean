#!/usr/bin/env bash
#MISE description="Run code formatters"
set -euo pipefail

mise run format:ruff
mise run format:imports
