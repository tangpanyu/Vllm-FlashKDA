#!/usr/bin/env bash
# Package the working tree, including ignored profiling data and Git history.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_NAME="$(basename -- "${REPO_ROOT}")"
OUTPUT="${1:-${REPO_ROOT}/../${REPO_NAME}-$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S).zip}"
OUTPUT="$(realpath -m -- "${OUTPUT}")"

if [[ "${OUTPUT}" == "${REPO_ROOT}" || "${OUTPUT}" == "${REPO_ROOT}/"* ]]; then
    echo "ZIP output must be outside the repository: ${OUTPUT}" >&2
    exit 1
fi
if [[ -e "${OUTPUT}" ]]; then
    echo "Output already exists; choose a new ZIP name: ${OUTPUT}" >&2
    exit 1
fi
for TOOL in zip unzip sha256sum; do
    command -v "${TOOL}" >/dev/null || { echo "Missing tool: ${TOOL}" >&2; exit 1; }
done
if [[ ! -f "${REPO_ROOT}/cutlass/include/cute/tensor.hpp" ]]; then
    echo "CUTLASS sources are missing; initialize the cutlass submodule first." >&2
    exit 1
fi

mkdir -p -- "$(dirname -- "${OUTPUT}")"
# Build the archive under a temporary name; publish only after CRC validation.
STAGING="$(mktemp -d "$(dirname -- "${OUTPUT}")/.flashkda-package.XXXXXX")"
trap 'rm -rf -- "${STAGING}"' EXIT
cd -- "${REPO_ROOT}/.."
echo "Packaging ${REPO_ROOT} (including NCU reports, artifacts, and .git)..."
# -y keeps the legacy snapshot's relative links to shared docs/CUTLASS/reports.
zip -q -r -y "${STAGING}/workspace.zip" "${REPO_NAME}" \
    -x '*/.cache/*' '*/__pycache__/*' '*/.pytest_cache/*' \
       '*/build/*' '*.egg-info/*' '*/.git/worktrees/*' '*/dist/*'
unzip -tq "${STAGING}/workspace.zip"
mv -- "${STAGING}/workspace.zip" "${OUTPUT}"
(
    cd -- "$(dirname -- "${OUTPUT}")"
    sha256sum -- "$(basename -- "${OUTPUT}")" > "${OUTPUT}.sha256"
)
du -h -- "${OUTPUT}"
echo "SHA256: ${OUTPUT}.sha256"
