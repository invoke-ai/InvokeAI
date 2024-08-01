from typing import Optional

import torch
from PIL import Image
from transformers.pipelines import ZeroShotObjectDetectionPipeline

from invokeai.backend.image_util.grounding_dino.detection_result import DetectionResult
from invokeai.backend.raw_model import RawModel


class GroundingDinoPipeline(RawModel):
    """A wrapper class for a ZeroShotObjectDetectionPipeline that makes it compatible with the model manager's memory
    management system.
    """

    def __init__(self, pipeline: ZeroShotObjectDetectionPipeline):
        self._pipeline = pipeline

    def detect(self, image: Image.Image, candidate_labels: list[str], threshold: float = 0.1) -> list[DetectionResult]:
        results = self._pipeline(image=image, candidate_labels=candidate_labels, threshold=threshold)
        assert results is not None
        results = [DetectionResult.model_validate(result) for result in results]
        return results

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        # HACK(ryand): The GroundingDinoPipeline does not work on MPS devices. We only allow it to be moved to CPU or
        # CUDA.
        if device is not None and device.type not in {"cpu", "cuda"}:
            device = None
        self._pipeline.model.to(device=device, dtype=dtype)
        self._pipeline.device = self._pipeline.model.device

    def calc_size(self) -> int:
        # HACK(ryand): Fix the circular import issue.
        from invokeai.backend.model_manager.load.model_util import calc_module_size

        return calc_module_size(self._pipeline.model)
