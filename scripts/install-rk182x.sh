#!/usr/bin/env bash
# Install and verify the RK182x environment from the bundled offline artifacts.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venvs/rk182x}"
VENDOR="${PROJECT_DIR}/vendor/wheels"
RKNN3_PACKAGE="${PROJECT_DIR}/vendor/rknn3/rknn3lite-cp311-aarch64.tar.gz"
NUMPY="numpy-1.26.4-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
OPENCV="opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"

fail() { echo "ERROR: $*" >&2; exit 1; }

case "$(uname -m)" in
    aarch64 | arm64) ;;
    *) fail "this environment targets the RK3576 board (aarch64 Linux); found $(uname -m)" ;;
esac

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || fail "${PYTHON_BIN} was not found; install Python 3.11 first"
PYTHON_TAG="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[ "${PYTHON_TAG}" = "3.11" ] || fail "the bundled packages require CPython 3.11; found ${PYTHON_TAG}"
"${PYTHON_BIN}" -c 'import venv' >/dev/null 2>&1 || fail "the venv module is missing; run: sudo apt install -y python3-venv"

for artifact in "${VENDOR}/${NUMPY}" "${VENDOR}/${OPENCV}" "${RKNN3_PACKAGE}"; do
    [ -f "${artifact}" ] || fail "a bundled dependency is missing: ${artifact}"
done

echo "verifying the shipped models, dependencies and sample clip"
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/verify-checksums.py" --quiet || fail "the shipped files do not match checksums.sha256"

if [ ! -d "${VENV_DIR}" ]; then
    echo "creating ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
if [ -f "${VENV_DIR}/pyvenv.cfg" ] && grep -qi '^include-system-site-packages *= *true' "${VENV_DIR}/pyvenv.cfg"; then
    echo "recreating ${VENV_DIR} without system site-packages"
    rm -rf "${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"
"${VENV_PYTHON}" -m pip --version >/dev/null 2>&1 || fail "pip is missing inside ${VENV_DIR}"

# rknn3-toolkit-lite declares numpy and transformers; transformers is only used by the LLM helper
# (rknn3lite/api/rknn3_lite_llm.py), which this project never imports
echo "installing NumPy and OpenCV from vendor/"
"${VENV_PYTHON}" -m pip install --quiet --no-index --no-deps --force-reinstall \
    "${VENDOR}/${NUMPY}" "${VENDOR}/${OPENCV}"

VENV_SITE="$("${VENV_PYTHON}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "installing the RKNN3 binding into ${VENV_SITE}"
tar xzf "${RKNN3_PACKAGE}" -C "${VENV_SITE}"

echo
"${VENV_PYTHON}" "${PROJECT_DIR}/scripts/check-rk182x.py"
echo
echo "done. Run a clip with: bash scripts/run-rk182x.sh [video]"
