#!/usr/bin/env bash
# Install and verify the RK3576 built-in NPU environment. No network access is needed: the
# repository carries the RKNNLite2 2.3.2 wheel and the board's own numpy/OpenCV/librknnrt are
# reused through a venv with --system-site-packages.
#
#   bash scripts/install-rk3576.sh
#
# Afterwards run:  bash scripts/run-rk3576.sh [video]
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venvs/rk3576}"
WHEEL="${PROJECT_DIR}/wheels/rknn_toolkit_lite2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"

fail() { echo "ERROR: $*" >&2; exit 1; }

case "$(uname -m)" in
    aarch64 | arm64) ;;
    *) fail "this environment targets RK3576 (aarch64 Linux); found $(uname -m)" ;;
esac

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || fail "${PYTHON_BIN} was not found; install Python 3.11 first"
PYTHON_TAG="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[ "${PYTHON_TAG}" = "3.11" ] || fail "the bundled RKNNLite2 wheel is CPython 3.11; found ${PYTHON_TAG} (set PYTHON_BIN=/path/to/python3.11)"
"${PYTHON_BIN}" -c 'import venv' >/dev/null 2>&1 || fail "the venv module is missing; run: sudo apt install -y python3-venv"
[ -f "${WHEEL}" ] || fail "the bundled RKNNLite2 wheel is missing: ${WHEEL}"

echo "verifying the shipped model, wheel and sample clip against checksums.sha256"
(cd "${PROJECT_DIR}" && sha256sum -c --quiet checksums.sha256) || fail "checksums.sha256 does not match the shipped files"

if [ ! -d "${VENV_DIR}" ]; then
    echo "creating ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"
fi
VENV_PYTHON="${VENV_DIR}/bin/python"

echo "installing the bundled RKNNLite2 wheel into the venv"
"${VENV_PYTHON}" -m pip install --quiet --no-index --no-deps --force-reinstall "${WHEEL}"

if ! "${VENV_PYTHON}" -c 'import numpy, cv2' >/dev/null 2>&1; then
    echo "note: numpy or OpenCV is not importable from the venv - install them with"
    echo "      sudo apt install -y python3-numpy python3-opencv"
fi

echo
"${VENV_PYTHON}" "${PROJECT_DIR}/scripts/check-rk3576.py"
STATUS=$?
echo
if [ "${STATUS}" -eq 0 ]; then
    echo "done. Run a clip with: bash scripts/run-rk3576.sh [video]"
fi
exit "${STATUS}"