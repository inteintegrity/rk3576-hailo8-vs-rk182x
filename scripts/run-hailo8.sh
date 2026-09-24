#!/usr/bin/env bash
# Run Hailo-8 on a clip, through the environment the install script created.
#
#   bash scripts/run-hailo8.sh                      # the bundled sample clip
#   bash scripts/run-hailo8.sh /path/to/clip.mp4    # any clip
#   bash scripts/run-hailo8.sh video/test.mp4 --depth 8     # extra flags pass through
#
# There is deliberately no fallback to the system Python: if .venvs/hailo8 is missing, install first.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venvs/hailo8}"
VENV_PYTHON="${VENV_DIR}/bin/python"

if [ ! -x "${VENV_PYTHON}" ]; then
    echo "ERROR: ${VENV_DIR} does not exist." >&2
    echo "       Install the environment first:  bash scripts/install-hailo8.sh" >&2
    exit 1
fi

VIDEO="${1:-${PROJECT_DIR}/video/test.mp4}"
if [ "$#" -gt 0 ]; then shift; fi

exec "${VENV_PYTHON}" "${PROJECT_DIR}/Hailo/Hailo8/run_video_inference.py" \
    --hef "${PROJECT_DIR}/model/Hailo/yolo26n_hailo8_official.hef" \
    --video "${VIDEO}" \
    --out-dir "${PROJECT_DIR}/out/hailo8" \
    "$@"