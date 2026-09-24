#!/usr/bin/env bash
# Run the RK3576 built-in NPU on a clip, through the environment the install script created.
#
#   bash scripts/run-rk3576.sh                      # the bundled sample clip
#   bash scripts/run-rk3576.sh /path/to/clip.mp4    # any clip (letterboxed to 640x640)
#   bash scripts/run-rk3576.sh video/test.mp4 --max-frames 100 --conf 0.3   # extra flags pass through
#
# There is deliberately no fallback to the system Python: if .venvs/rk3576 is missing, install first.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venvs/rk3576}"
VENV_PYTHON="${VENV_DIR}/bin/python"

if [ ! -x "${VENV_PYTHON}" ]; then
    echo "ERROR: ${VENV_DIR} does not exist." >&2
    echo "       Install the environment first:  bash scripts/install-rk3576.sh" >&2
    exit 1
fi

VIDEO="${1:-${PROJECT_DIR}/video/test.mp4}"
if [ "$#" -gt 0 ]; then shift; fi

exec "${VENV_PYTHON}" "${PROJECT_DIR}/rk3576/run_video_inference.py" \
    --model "${PROJECT_DIR}/model/rk3576/yolo26n_rk3576_int8.rknn" \
    --video "${VIDEO}" \
    --out-dir "${PROJECT_DIR}/out/rk3576" \
    "$@"