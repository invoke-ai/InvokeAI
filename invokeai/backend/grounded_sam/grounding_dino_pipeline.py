from typing import Optional

import torch
from transformers.pipelines import ZeroShotObjectDetectionPipeline


class GroundingDinoPipeline:
    """A wrapper class for a ZeroShotObjectDetectionPipeline that makes it compatible with the model manager's memory
    management system.
    """

    def __init__(self, pipeline: ZeroShotObjectDetectionPipeline):
        self._pipeline = pipeline

    def __call__(self, *args, **kwargs):
        return self._pipeline(*args, **kwargs)

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> "GroundingDinoPipeline":
        self._pipeline.model.to(device=device, dtype=dtype)
        self._pipeline.device = self._pipeline.model.device
        return self

    def calc_size(self) -> int:
        # HACK(ryand): Fix the circular import issue.
        from invokeai.backend.model_manager.load.model_util import calc_module_size

        return calc_module_size(self._pipeline.model)
