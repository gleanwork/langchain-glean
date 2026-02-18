#!/usr/bin/env bash
#MISE description="Run unit tests"
#MISE alias="tests"
set -euo pipefail

uv run pytest --disable-socket --allow-unix-socket -v --tb=auto -rA --durations=10 -p no:logging tests/
