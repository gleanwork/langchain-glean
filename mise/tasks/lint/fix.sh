#!/usr/bin/env bash
#MISE description="Run lint autofixers"
set -euo pipefail

mise run lint:fix:ruff
mise run format
