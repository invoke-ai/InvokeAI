#!/bin/bash
set -e

source "${CONDA_PREFIX}/etc/profile.d/conda.sh"
conda activate "${PROJECT_NAME}"

python scripts/invoke.py \
  ${@:---web --host=0.0.0.0}
