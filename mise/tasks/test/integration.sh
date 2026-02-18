#!/usr/bin/env bash
#MISE description="Run integration tests"
#MISE alias=["integration_test", "integration_tests"]
set -euo pipefail

uv run pytest -v --tb=auto -rA --durations=10 -p no:logging tests/integration_tests/
