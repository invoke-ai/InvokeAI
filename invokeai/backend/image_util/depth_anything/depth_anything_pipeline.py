from typing import Optional, cast

import torch
from PIL import Image
from transformers.pipelines import DepthEstimationPipeline

from invokeai.backend.raw_model import RawModel


class DepthAnythingPipeline(RawModel):
    """Custom wrapper for the Depth Estimation pipeline from transformers adding compatibility
    for Invoke's Model Management System"""

    def __init__(self, pipeline: DepthEstimationPipeline) -> None:
        self.pipeline = pipeline

    def generate_depth(self, image: Image.Image, resolution: int = 512):
        image_width, image_height = image.size
        depth_map = self.pipeline(image)["depth"]
        depth_map = cast(Image.Image, depth_map)

        new_height = int(image_height * (resolution / image_width))
        depth_map = depth_map.resize((resolution, new_height))
        return depth_map

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        pass

    def calc_size(self) -> int:
        from invokeai.backend.model_manager.load.model_util import calc_module_size

        return calc_module_size(self.pipeline.model)
