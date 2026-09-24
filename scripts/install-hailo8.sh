#!/usr/bin/env bash
# Install and verify the Hailo-8 environment.
#
#   bash scripts/install-hailo8.sh                              # use the board's existing HailoRT
#   HAILORT_WHEEL=/path/to/hailort-4.23.0-cp311-cp311-linux_aarch64.whl bash scripts/install-hailo8.sh
#
# This repository deliberately does not ship a HailoRT wheel (the Python bindings come from Hailo's
# Developer Zone and are not redistributed here), so the default route is "whatever HailoRT 4.23 is
# already installed on the board". A wheel installs only the Python bindings - the HailoRT library,
# the PCIe driver and hailortcli must still be installed on the board itself.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venvs/hailo8}"
HAILORT_WHEEL="${HAILORT_WHEEL:-}"

fail() { echo "ERROR: $*" >&2; exit 1; }

case "$(uname -m)" in
    aarch64 | arm64) ;;
    *) fail "this environment targets the RK3576 board (aarch64 Linux); found $(uname -m)" ;;
esac

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || fail "${PYTHON_BIN} was not found; install Python 3.11 first"
PYTHON_TAG="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[ "${PYTHON_TAG}" = "3.11" ] || fail "HailoRT 4.23 wheels are CPython 3.11; found ${PYTHON_TAG} (set PYTHON_BIN=/path/to/python3.11)"
"${PYTHON_BIN}" -c 'import venv' >/dev/null 2>&1 || fail "the venv module is missing; run: sudo apt install -y python3-venv"

echo "verifying the shipped model and sample clip against checksums.sha256"
(cd "${PROJECT_DIR}" && sha256sum -c --quiet checksums.sha256) || fail "checksums.sha256 does not match the shipped files"

if [ ! -d "${VENV_DIR}" ]; then
    echo "creating ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"
fi
VENV_PYTHON="${VENV_DIR}/bin/python"

if [ -n "${HAILORT_WHEEL}" ]; then
    [ -f "${HAILORT_WHEEL}" ] || fail "HAILORT_WHEEL points at a missing file: ${HAILORT_WHEEL}"
    echo "installing ${HAILORT_WHEEL##*/} into the venv"
    "${VENV_PYTHON}" -m pip install --quiet --no-index --no-deps --force-reinstall "${HAILORT_WHEEL}"
elif "${VENV_PYTHON}" -c 'import hailo_platform' >/dev/null 2>&1; then
    echo "using the HailoRT already installed on this board"
else
    fail "HailoRT is not installed on this board and no wheel was given.
       Install HailoRT 4.23 for this board (driver + library + hailortcli) from Hailo's
       Developer Zone, or pass the Python wheel explicitly:
         HAILORT_WHEEL=/path/to/hailort-4.23.0-cp311-cp311-linux_aarch64.whl bash scripts/install-hailo8.sh"
fi

if ! "${VENV_PYTHON}" -c 'import numpy, cv2' >/dev/null 2>&1; then
    echo "note: numpy or OpenCV is not importable from the venv - install them with"
    echo "      sudo apt install -y python3-numpy python3-opencv"
fi

echo
"${VENV_PYTHON}" "${PROJECT_DIR}/scripts/check-hailo8.py"
STATUS=$?
echo
if [ "${STATUS}" -eq 0 ]; then
    echo "done. Run a clip with: bash scripts/run-hailo8.sh [video]"
fi
exit "${STATUS}"