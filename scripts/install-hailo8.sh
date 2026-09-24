#!/usr/bin/env bash
# Install and verify the Hailo-8 environment.
#
#   bash scripts/install-hailo8.sh
#
# Everything comes from vendor/ into .venvs/hailo8, offline: NumPy, OpenCV and the HailoRT 4.23
# Python binding all ship as aarch64 wheels in this repository. The venv is self-contained, so the
# runner never needs PYTHONPATH or LD_LIBRARY_PATH.
#
# The board keeps its own kernel driver and HailoRT libraries: this script installs nothing at the
# system level, replaces no driver and does not reload modules. It only makes the Python side (the
# binding) available to the venv. If the Hailo-8 card is not in the M.2 slot the check will say so;
# everything else still installs.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venvs/hailo8}"
VENDOR="${PROJECT_DIR}/vendor/wheels"
NUMPY="numpy-1.26.4-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
OPENCV="opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
HAILORT="hailort-4.23.0-cp311-cp311-linux_aarch64.whl"
# the binding declares these; netifaces is only used for Ethernet Hailo devices (see vendor/README.md)
ARGCOMPLETE="argcomplete-3.7.2-py3-none-any.whl"
CONTEXTLIB2="contextlib2-21.6.0-py2.py3-none-any.whl"
FUTURE="future-1.0.0-py3-none-any.whl"
NETADDR="netaddr-1.3.0-py3-none-any.whl"

fail() { echo "ERROR: $*" >&2; exit 1; }

case "$(uname -m)" in
    aarch64 | arm64) ;;
    *) fail "this environment targets the RK3576 board (aarch64 Linux); found $(uname -m)" ;;
esac

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || fail "${PYTHON_BIN} was not found; install Python 3.11 first"
PYTHON_TAG="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[ "${PYTHON_TAG}" = "3.11" ] || fail "the bundled wheels are CPython 3.11; found ${PYTHON_TAG} (set PYTHON_BIN=/path/to/python3.11)"
"${PYTHON_BIN}" -c 'import venv' >/dev/null 2>&1 || fail "the venv module is missing; run: sudo apt install -y python3-venv"

for wheel in "${NUMPY}" "${OPENCV}" "${HAILORT}" "${ARGCOMPLETE}" "${CONTEXTLIB2}" "${FUTURE}" "${NETADDR}"; do
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

echo "installing the full dependency set from vendor/"
"${VENV_PYTHON}" -m pip install --quiet --no-index --no-deps --force-reinstall \
    "${VENDOR}/${NUMPY}" "${VENDOR}/${OPENCV}" "${VENDOR}/${HAILORT}" \n    "${VENDOR}/${ARGCOMPLETE}" "${VENDOR}/${CONTEXTLIB2}" "${VENDOR}/${FUTURE}" "${VENDOR}/${NETADDR}"

echo
"${VENV_PYTHON}" "${PROJECT_DIR}/scripts/check-hailo8.py"
STATUS=$?
echo
if [ "${STATUS}" -eq 0 ]; then
    echo "done. Run a clip with: bash scripts/run-hailo8.sh [video]"
fi
exit "${STATUS}"
