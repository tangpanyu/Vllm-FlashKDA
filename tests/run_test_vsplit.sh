#!/usr/bin/env bash
set -euo pipefail

# Focused split/cluster matrix.  Keep this separate from run_test_full.sh so
# feature validation does not enlarge the repository's expensive full suite.
python -m pytest tests/test_vsplit.py -x -q -s
