#!/usr/bin/env bash
#MISE description="Run all tests and lint fixes"
set -euo pipefail

mise run test
mise run lint:fix
mise run lint:mypy
