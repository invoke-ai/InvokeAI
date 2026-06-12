#!/bin/sh
if [ -z "$1" ] ; then
    echo "Usage: $0 GPU"
    echo "   where GPU is an extra defined in pyproject.toml [cpu, cuda, rocm]"
    exit 1
fi
set -ex
sh .devcontainer/run-pnpm-i.sh invokeai/frontend/web docs
uv sync --frozen --no-progress --extra="$1" --extra dev --extra test --extra docs --extra dist
# collect_env shows whether torch is set up correctly.
uv run --no-sync python -m torch.utils.collect_env
