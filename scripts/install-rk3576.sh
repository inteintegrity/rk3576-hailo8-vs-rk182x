#!/usr/bin/env bash
# Install and verify the RK3576 built-in NPU environment.
#
#   bash scripts/install-rk3576.sh
#
# Everything is installed from vendor/ into .venvs/rk3576, offline: the RKNNLite2 binding, NumPy
# and OpenCV all ship as aarch64 wheels in this repository. The venv is self-contained (it does
# not inherit the system site-packages), so the runner never needs PYTHONPATH or LD_LIBRARY_PATH.
# The board's own librknnrt.so and its rknpu driver are used as they are - nothing here installs,
# replaces or reloads a kernel driver.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venvs/rk3576}"
VENDOR="${PROJECT_DIR}/vendor/wheels"
NUMPY="numpy-1.26.4-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
OPENCV="opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
RKNNLITE="rknn_toolkit_lite2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"

fail() { echo "ERROR: $*" >&2; exit 1; }

case "$(uname -m)" in
    aarch64 | arm64) ;;
    *) fail "this environment targets RK3576 (aarch64 Linux); found $(uname -m)" ;;
esac

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || fail "${PYTHON_BIN} was not found; install Python 3.11 first"
PYTHON_TAG="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[ "${PYTHON_TAG}" = "3.11" ] || fail "the bundled wheels are CPython 3.11; found ${PYTHON_TAG} (set PYTHON_BIN=/path/to/python3.11)"
"${PYTHON_BIN}" -c 'import venv' >/dev/null 2>&1 || fail "the venv module is missing; run: sudo apt install -y python3-venv"

for wheel in "${NUMPY}" "${OPENCV}" "${RKNNLITE}"; do
    [ -f "${VENDOR}/${wheel}" ] || fail "a bundled wheel is missing: ${VENDOR}/${wheel}"
done

echo "verifying the shipped models, wheels and sample clip against checksums.sha256"
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/verify-checksums.py" --quiet || fail "the shipped files do not match checksums.sha256"

if [ ! -d "${VENV_DIR}" ]; then
    echo "creating ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
# the environment must be self-contained: a venv made by an earlier revision inherited the system
# site-packages, and the runner must not depend on that
if [ -f "${VENV_DIR}/pyvenv.cfg" ] && grep -qi '^include-system-site-packages *= *true' "${VENV_DIR}/pyvenv.cfg"; then
    echo "${VENV_DIR} still inherits the system site-packages; recreating it self-contained"
    rm -rf "${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
VENV_PYTHON="${VENV_DIR}/bin/python"
"${VENV_PYTHON}" -m pip --version >/dev/null 2>&1 || fail "pip is missing inside ${VENV_DIR}; run: sudo apt install -y python3-venv python3-pip"

echo "installing NumPy, OpenCV and the RKNNLite2 binding from vendor/"
"${VENV_PYTHON}" -m pip install --quiet --no-index --no-deps --force-reinstall \
    "${VENDOR}/${NUMPY}" "${VENDOR}/${OPENCV}" \

echo
"${VENV_PYTHON}" "${PROJECT_DIR}/scripts/check-rk3576.py"
STATUS=$?
echo
if [ "${STATUS}" -eq 0 ]; then
    echo "done. Run a clip with: bash scripts/run-rk3576.sh [video]"
fi
exit "${STATUS}"
