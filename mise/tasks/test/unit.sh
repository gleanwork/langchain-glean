#!/usr/bin/env bash
#MISE description="Run unit tests"
#MISE alias=["unit_test", "unit_tests"]
set -euo pipefail

uv run pytest -v --tb=auto -rA --durations=10 -p no:logging tests/unit_tests/
