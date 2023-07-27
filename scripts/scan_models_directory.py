#!/usr/bin/env python

"""
Scan the models directory and print out a new models.yaml
"""

import os
import sys
import argparse

from pathlib import Path
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description="Model directory scanner")
    parser.add_argument("models_directory")
    parser.add_argument(
        "--all-models",
        default=False,
        action="store_true",
        help="If true, then generates stanzas for all models; otherwise just diffusers",
    )

    args = parser.parse_args()
    directory = args.models_directory

    conf = OmegaConf.create()
    conf["_version"] = "3.0.0"

    for root, dirs, files in os.walk(directory):
        parents = root.split("/")
        subpaths = parents[parents.index("models") + 1 :]
        if len(subpaths) < 2:
            continue
        base, model_type, *_ = subpaths

        if args.all_models or model_type == "diffusers":
            for d in dirs:
                conf[f"{base}/{model_type}/{d}"] = dict(
                    path=os.path.join(root, d),
                    description=f"{model_type} model {d}",
                    format="folder",
                    base=base,
                )

            for f in files:
                basename = Path(f).stem
                format = Path(f).suffix[1:]
                conf[f"{base}/{model_type}/{basename}"] = dict(
                    path=os.path.join(root, f),
                    description=f"{model_type} model {basename}",
                    format=format,
                    base=base,
                )

    OmegaConf.save(config=dict(sorted(conf.items())), f=sys.stdout)


if __name__ == "__main__":
    main()
