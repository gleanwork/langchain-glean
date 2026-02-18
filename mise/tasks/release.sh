#!/usr/bin/env bash
#MISE description="Bump version and create a new tag (use DRY_RUN=true for preview)"
set -euo pipefail

DRY_RUN="${DRY_RUN:-false}"
CZ="uv run python -m commitizen"
if [ "$DRY_RUN" = "true" ]; then
  $CZ bump --dry-run
  $CZ changelog --dry-run
else
  $CZ bump --yes
  $CZ changelog
fi
