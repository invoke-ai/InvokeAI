#!/bin/bash
if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    . "/opt/conda/etc/profile.d/conda.sh"
    CONDA_CHANGEPS1=false conda activate invokeai
fi

#cd /InvokeAI/
conda activate invokeai

python3 scripts/invoke.py --web --host 0.0.0.0 --port 9090
