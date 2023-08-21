# Copyright (c) 2023 Lincoln Stein and the InvokeAI Team
"""
Module for probing a Stable Diffusion model and returning
its base type, model type, format and variant.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

import torch
import safetensors.torch

from invokeai.backend.model_management.models.base import (
    read_checkpoint_meta
)
import invokeai.configs.model_probe_templates as templates

from .config import (
    ModelType,
    BaseModelType,
    ModelVariantType,
    ModelFormat,
    SchedulerPredictionType
)


@dataclass
class ModelProbeInfo(object):
    model_type: ModelType
    base_type: BaseModelType
    variant_type: ModelVariantType
    prediction_type: SchedulerPredictionType
    format: ModelFormat

class ModelProbe(object):
    """
    Class to probe a checkpoint, safetensors or diffusers folder.
    """

    def __init__(self):
        pass

    @classmethod
    def heuristic_probe(
            cls,
            model: Path,
            prediction_type_helper: Optional[Callable[[Path], SchedulerPredictionType]] = None,
    ) -> ModelProbeInfo:
        """
        Probe model located at path and return ModelProbeInfo object.
        A Callable may be passed to return the SchedulerPredictionType.
        """
        pass
