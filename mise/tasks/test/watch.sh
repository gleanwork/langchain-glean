#!/usr/bin/env bash
#MISE description="Run tests in watch mode"
set -euo pipefail

uv run ptw --snapshot-update --now . -- -vv --tb=auto -rA --durations=10 -p no:logging tests/
