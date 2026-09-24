#!/usr/bin/env bash
# Install and verify the RK182x environment (RKNN3 runtime).
#
#   bash scripts/install-rk182x.sh                                   # use the board's existing runtime
#   RKNN3_WHEEL=/path/to/rknn3_toolkit_lite-1.0.4-cp311-*.whl bash scripts/install-rk182x.sh
#
# This repository does not ship an RKNN3 wheel (it comes with the RK182x SDK and must match the
# runtime installed on the board). If neither the board's runtime nor RKNN3_WHEEL is available,
# this script says so here - it never lets you find out at inference time.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venvs/rk182x}"
RKNN3_WHEEL="${RKNN3_WHEEL:-}"

fail() { echo "ERROR: $*" >&2; exit 1; }

case "$(uname -m)" in
    aarch64 | arm64) ;;
    *) fail "this environment targets the RK3576 board (aarch64 Linux); found $(uname -m)" ;;
esac

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || fail "${PYTHON_BIN} was not found; install Python 3.11 first"
PYTHON_TAG="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[ "${PYTHON_TAG}" = "3.11" ] || fail "the RKNN3 board wheels are CPython 3.11; found ${PYTHON_TAG} (set PYTHON_BIN=/path/to/python3.11)"
"${PYTHON_BIN}" -c 'import venv' >/dev/null 2>&1 || fail "the venv module is missing; run: sudo apt install -y python3-venv"

echo "verifying the shipped model, weight and sample clip against checksums.sha256"
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/verify-checksums.py" --quiet || fail "the shipped files do not match checksums.sha256"

if [ ! -d "${VENV_DIR}" ]; then
    echo "creating ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"
fi
VENV_PYTHON="${VENV_DIR}/bin/python"

if [ -n "${RKNN3_WHEEL}" ]; then
    [ -f "${RKNN3_WHEEL}" ] || fail "RKNN3_WHEEL points at a missing file: ${RKNN3_WHEEL}"
    echo "installing ${RKNN3_WHEEL##*/} into the venv"
    "${VENV_PYTHON}" -m pip install --quiet --no-index --no-deps --force-reinstall "${RKNN3_WHEEL}"
elif "${VENV_PYTHON}" -c 'import rknn3lite.api.rknn3_lite' >/dev/null 2>&1; then
    echo "using the RKNN3 runtime already installed on this board"
else
    fail "the RKNN3 runtime is not installed on this board and this repository has no wheel for it.
       Install the RK182x SDK (runtime 1.0.4 for this board) so that 'rknn3lite' imports, or point
       at the matching wheel explicitly - toolkit and runtime versions must be the same:
         RKNN3_WHEEL=/path/to/rknn3_toolkit_lite-1.0.4-cp311-*.whl bash scripts/install-rk182x.sh
       If the SDK keeps 'rknn3lite' outside site-packages, make it visible to the venv with
       PYTHONPATH=<sdk python dir> before running this script."
fi

if ! "${VENV_PYTHON}" -c 'import numpy, cv2' >/dev/null 2>&1; then
    echo "note: numpy or OpenCV is not importable from the venv - install them with"
    echo "      sudo apt install -y python3-numpy python3-opencv"
fi

echo
"${VENV_PYTHON}" "${PROJECT_DIR}/scripts/check-rk182x.py"
STATUS=$?
echo
if [ "${STATUS}" -eq 0 ]; then
    echo "done. Run a clip with: bash scripts/run-rk182x.sh [video]"
fi
exit "${STATUS}"