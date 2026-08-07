#!/usr/bin/env bash
set -euo pipefail

# Profile only the 5-warp, non-cluster V-split K2 specialization.
# The extension must already be rebuilt from experiment/vsplit-5warp.

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-$(python -c 'import sys; print(sys.executable)')}"
NCU_BIN="${NCU_BIN:-$(type -P ncu)}"
REPORT="${REPORT:-h100_b16_t8192_h96_split5_noncluster}"
REPORT="${REPORT%.ncu-rep}"

if [[ -z "${NCU_BIN}" ]]; then
    echo "ncu was not found; set NCU_BIN explicitly" >&2
    exit 1
fi

echo "Python: ${PYTHON_BIN}"
echo "NCU:    ${NCU_BIN}"
echo "Report: ${REPORT}.ncu-rep"

sudo -E "${NCU_BIN}" \
    --set full \
    --replay-mode application \
    --app-replay-match grid \
    --app-replay-mode strict \
    --cache-control none \
    --kernel-name-base function \
    --kernel-name 'regex:.*flash_kda_fwd_recurrence.*' \
    --launch-count 1 \
    --force-overwrite \
    --export "${REPORT}" \
    "${PYTHON_BIN}" benchmarks/bench_vsplit.py \
        --mode profile \
        --case pure_n16_t8192_h96 \
        --profile-variant split \
        --profile-launches 1

echo "Completed: ${REPORT}.ncu-rep"
