#!/usr/bin/env bash
# Run RK182x on a clip, through the environment the install script created.
#
#   bash scripts/run-rk182x.sh                      # the bundled sample clip
#   bash scripts/run-rk182x.sh /path/to/clip.mp4    # any clip
#   bash scripts/run-rk182x.sh video/test.mp4 --core-mask 0x3   # extra flags pass through
#
# There is deliberately no fallback to the system Python: if .venvs/rk182x is missing, install first.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venvs/rk182x}"
VENV_PYTHON="${VENV_DIR}/bin/python"

if [ ! -x "${VENV_PYTHON}" ]; then
    echo "ERROR: ${VENV_DIR} does not exist." >&2
    echo "       Install the environment first:  bash scripts/install-rk182x.sh" >&2
    exit 1
fi

VIDEO="${1:-${PROJECT_DIR}/video/test.mp4}"
if [ "$#" -gt 0 ]; then shift; fi

exec "${VENV_PYTHON}" "${PROJECT_DIR}/rk182x/rk1820/run_video_inference.py" \
    --model "${PROJECT_DIR}/model/rk1820/yolo26n_rk1820_int8.rknn" \
    --weight "${PROJECT_DIR}/model/rk1820/yolo26n_rk1820_int8.weight" \
    --video "${VIDEO}" \
    --out-dir "${PROJECT_DIR}/out/rk182x" \
    "$@"