from typing import Optional

import torch
from PIL import Image
from transformers.pipelines import DepthEstimationPipeline

from invokeai.backend.raw_model import RawModel


class DepthAnythingPipeline(RawModel):
    """Custom wrapper for the Depth Estimation pipeline from transformers adding compatibility
    for Invoke's Model Management System"""

    def __init__(self, pipeline: DepthEstimationPipeline) -> None:
        self._pipeline = pipeline

    def generate_depth(self, image: Image.Image) -> Image.Image:
        depth_map = self._pipeline(image)["depth"]
        assert isinstance(depth_map, Image.Image)
        return depth_map

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        if device is not None and device.type not in {"cpu", "cuda"}:
            device = None
        self._pipeline.model.to(device=device, dtype=dtype)
        self._pipeline.device = self._pipeline.model.device

    def calc_size(self) -> int:
        from invokeai.backend.model_manager.load.model_util import calc_module_size

        return calc_module_size(self._pipeline.model)
