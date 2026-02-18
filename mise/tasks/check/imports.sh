#!/usr/bin/env bash
#MISE description="Check imports"
set -euo pipefail

find langchain_glean -name '*.py' | grep -v 'docs/' | xargs uv run python ./scripts/check_imports.py
