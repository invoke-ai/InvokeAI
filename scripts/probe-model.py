#!/bin/env python

"""Little command-line utility for probing a model on disk."""

import argparse
import json
import sys
from pathlib import Path

from invokeai.backend.model_manager import InvalidModelException, ModelProbe, SchedulerPredictionType


def helper(model_path: Path):
    print('Warning: guessing "v_prediction" SchedulerPredictionType', file=sys.stderr)
    return SchedulerPredictionType.VPrediction


parser = argparse.ArgumentParser(description="Probe model type")
parser.add_argument(
    "model_path",
    type=Path,
    nargs="+",
)
args = parser.parse_args()

for path in args.model_path:
    try:
        info = ModelProbe().probe(path, helper)
        print(f"{path}:{json.dumps(info.dict(), sort_keys=True, indent=4)}")
    except InvalidModelException as exc:
        print(exc)
