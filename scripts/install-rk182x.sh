#!/usr/bin/env bash
# Install and verify the RK182x environment.
#
#   bash scripts/install-rk182x.sh
#
# NumPy and OpenCV come from vendor/wheels. The RKNN3 binding (rknn3lite) has no wheel in this
# repository, so this script turns the RKNN3 Python environment that is already on the board into a
# portable package - vendor/rknn3/rknn3lite-cp311-aarch64.tar.gz - and installs it into
# .venvs/rk182x. The runner therefore never needs PYTHONPATH, and the same tarball can be carried to
# another board. The board's kernel driver and runtime libraries are used as they are: nothing here
# installs, replaces or reloads a driver.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venvs/rk182x}"
VENDOR="${PROJECT_DIR}/vendor/wheels"
RKNN3_DIR="${PROJECT_DIR}/vendor/rknn3"
RKNN3_PACKAGE="${RKNN3_DIR}/rknn3lite-cp311-aarch64.tar.gz"
NUMPY="numpy-1.26.4-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
OPENCV="opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"

# where the board's RKNN3 Python environment can live; RKNN3_SOURCE is only an override for unusual
# layouts, never something the user has to set for a normal board
RKNN3_SOURCES=(
    "${RKNN3_SOURCE:-}"
    "${HOME}/rk1820_yolo/rknn3_env"
    "${HOME}/rknn3_env"
    "/opt/rknn3/rknn3_env"
)

fail() { echo "ERROR: $*" >&2; exit 1; }

case "$(uname -m)" in
    aarch64 | arm64) ;;
    *) fail "this environment targets the RK3576 board (aarch64 Linux); found $(uname -m)" ;;
esac

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || fail "${PYTHON_BIN} was not found; install Python 3.11 first"
PYTHON_TAG="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[ "${PYTHON_TAG}" = "3.11" ] || fail "the bundled wheels are CPython 3.11; found ${PYTHON_TAG} (set PYTHON_BIN=/path/to/python3.11)"
"${PYTHON_BIN}" -c 'import venv' >/dev/null 2>&1 || fail "the venv module is missing; run: sudo apt install -y python3-venv"

for wheel in "${NUMPY}" "${OPENCV}"; do
    [ -f "${VENDOR}/${wheel}" ] || fail "a bundled wheel is missing: ${VENDOR}/${wheel}"
done

echo "verifying the shipped models, wheels and sample clip against checksums.sha256"
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/verify-checksums.py" --quiet || fail "the shipped files do not match checksums.sha256"

# --- the RKNN3 binding: reuse the portable package, or build it from the board's environment -----
locate_rknn3_package() {
    local source candidate
    for source in "${RKNN3_SOURCES[@]}"; do
        [ -n "${source}" ] || continue
        for candidate in "${source}/rknn3lite" "${source}"/lib/python3.11/site-packages/rknn3lite; do
            if [ -d "${candidate}" ]; then
                printf '%s' "${candidate}"
                return 0
            fi
        done
        [ -d "${source}" ] && {
            candidate="$(find "${source}" -maxdepth 3 -type d -name rknn3lite 2>/dev/null | head -1)"
            [ -n "${candidate}" ] && { printf '%s' "${candidate}"; return 0; }
        }
    done
    "${PYTHON_BIN}" -c 'import rknn3lite, os; print(os.path.dirname(rknn3lite.__file__))' 2>/dev/null && return 0
    return 1
}

if [ ! -f "${RKNN3_PACKAGE}" ]; then
    PACKAGE_DIR="$(locate_rknn3_package || true)"
    [ -n "${PACKAGE_DIR}" ] || fail "no RKNN3 Python environment found on this board.
       Looked for rknn3lite under: ${RKNN3_SOURCES[*]}
       Install the RK182x SDK so that rknn3lite imports, or point at its directory:
         RKNN3_SOURCE=/path/to/rknn3_env bash scripts/install-rk182x.sh"
    echo "packing ${PACKAGE_DIR} into a portable offline package:"
    echo "  ${RKNN3_PACKAGE}"
    mkdir -p "${RKNN3_DIR}"
    "${PYTHON_BIN}" - "${PACKAGE_DIR}" "${RKNN3_PACKAGE}" <<'PACK'
import sys, tarfile
from pathlib import Path

package_dir = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2])
parent = package_dir.parent
members = [package_dir]
for entry in sorted(parent.iterdir()):
    if entry == package_dir:
        continue
    name = entry.name
    if name.startswith("rknn3") or name.startswith("rknn3lite-"):
        members.append(entry)
with tarfile.open(output, "w:gz") as tar:
    for member in members:
        tar.add(member, arcname=member.name)
print("   packed:", ", ".join(member.name for member in members))
PACK
    "${PYTHON_BIN}" - "$RKNN3_PACKAGE" <<'SUMS'
import hashlib, sys
from pathlib import Path
package = Path(sys.argv[1])
digest = hashlib.sha256(package.read_bytes()).hexdigest()
Path(str(package) + ".sha256").write_text(f"{digest}  {package.name}\n", encoding="utf-8", newline="\n")
print(f"   sha256: {digest}")
SUMS
fi

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

echo "installing NumPy and OpenCV from vendor/"
"${VENV_PYTHON}" -m pip install --quiet --no-index --no-deps --force-reinstall \
    "${VENDOR}/${NUMPY}" "${VENDOR}/${OPENCV}"

VENV_SITE="$("${VENV_PYTHON}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "installing the RKNN3 binding into ${VENV_SITE}"
tar xzf "${RKNN3_PACKAGE}" -C "${VENV_SITE}"

echo
"${VENV_PYTHON}" "${PROJECT_DIR}/scripts/check-rk182x.py"
STATUS=$?
echo
if [ "${STATUS}" -eq 0 ]; then
    echo "done. Run a clip with: bash scripts/run-rk182x.sh [video]"
fi
exit "${STATUS}"
