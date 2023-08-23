#!/usr/bin/env python
"""
Model template creator.

Scan a tree of checkpoint/safetensors/diffusers models and write out
a series of template .json file containing their  metadata for use
in fast model probing.
"""

import argparse
import json
import hashlib
import sys
import traceback
from pathlib import Path
from typing import List, Optional
import torch

from invokeai.backend.model_management.model_probe import ModelProbe, ModelProbeInfo
from invokeai.backend.model_management.model_search import ModelSearch
from invokeai.backend.model_manager import read_checkpoint_meta


class CreateTemplateScanner(ModelSearch):
    """Scan directory and create templates for each model found."""

    _dest: Path

    def __init__(self, directories: List[Path], dest: Path, **kwargs):  # noqa D401
        """Initialization routine.

        :param dest: Base of templates directory.
        """
        super().__init__(directories, **kwargs)
        self._dest = dest

    def on_model_found(self, model: Path):  # noqa D401
        """Called when a model is found during recursive search."""
        info: ModelProbeInfo = ModelProbe.probe(model)
        if not info:
            return
        self.write_template(model, info)

    def write_template(self, model: Path, info: ModelProbeInfo):
        """Write template for a checkpoint file."""
        dest_path = Path(self._dest,
                         "checkpoints" if model.is_file() else 'diffusers',
                         info.base_type.value,
                         info.model_type.value
                         )
        template: dict = self._make_checkpoint_template(model) \
            if model.is_file() \
            else self._make_diffusers_template(model)
        if not template:
            print(f"Could not create template for {model}, got {template}")
            return

        # sort the dict to avoid differences due to insertion order
        template = dict(sorted(template.items()))

        dest_path.mkdir(parents=True, exist_ok=True)
        meta = dict(
            base_type=info.base_type.value,
            model_type=info.model_type.value,
            variant=info.variant_type.value,
            template=template,
        )
        payload = json.dumps(meta)
        hash = hashlib.md5(payload.encode("utf-8")).hexdigest()
        try:
            dest_file = dest_path / f"{hash}.json"
            if not dest_file.exists():
                with open(dest_file, "w", encoding="utf-8") as f:
                    f.write(payload)
                    print(f"Template written out as {dest_file}")
        except OSError as e:
            print(f"An exception occurred while writing template: {str(e)}")

    def _make_checkpoint_template(self, model: Path) -> Optional[dict]:
        """Make template dict for a checkpoint-style model."""
        tmpl = None
        try:
            ckpt = read_checkpoint_meta(model)
            while "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            tmpl = {}
            for key, value in ckpt.items():
                if isinstance(value, torch.Tensor):
                    tmpl[key] = list(value.shape)
                elif isinstance(value, dict):  # handle one level of nesting - if more we should recurse
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, torch.Tensor):
                            tmpl[f"{key}.{subkey}"] = subvalue.shape
        except Exception as e:
            traceback.print_exception(e)
        return tmpl

    def _make_diffusers_template(self, model: Path) -> Optional[dict]:
        """
        Make template dict for a diffusers-style model.

        In case of a pipeline, template keys will be 'unet', 'text_encoder', 'text_encoder_2' and 'vae'.
        In case of another folder-style model, the template will simply contain the contents of config.json.
        """
        tmpl = None
        if (model / "model_index.json").exists():  # a pipeline
            tmpl = {}
            for subdir in ['unet', 'text_encoder', 'vae', 'text_encoder_2']:
                config = model / subdir / "config.json"
                try:
                    tmpl[subdir] = self._read_config(config)
                except FileNotFoundError:
                    pass
        elif (model / "learned_embeds.bin").exists():  # concepts model
            return self._make_checkpoint_template(model / "learned_embeds.bin")
        else:
            config = model / "config.json"
            try:
                tmpl = self._read_config(config)
            except FileNotFoundError:
                pass
        return tmpl

    def _read_config(self, config: Path) -> dict:
        with open(config, 'r', encoding='utf-8') as f:
            return {x: y for x, y in json.load(f).items() if not x.startswith("_")}

    def on_search_completed(self):
        """Not used."""
        pass

    def on_search_started(self):
        """Not used."""
        pass


parser = argparse.ArgumentParser(
    description="Scan the provided path recursively and create .json templates for all models found.",
)
parser.add_argument("--scan",
                    type=Path,
                    help="Path to recursively scan for models"
                    )
parser.add_argument("--out",
                    type=Path,
                    dest="outdir",
                    default=Path(__file__).resolve().parents[1] / "invokeai/configs/model_probe_templates",
                    help="Destination for templates",
                    )

opt = parser.parse_args()
scanner = CreateTemplateScanner([opt.scan], dest=opt.outdir)
scanner.search()
