import pathlib
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import pipeline
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

    def generate_depth_with_unit(self, image: Image.Image) -> tuple[Image.Image, np.ndarray]:
        """The uint8 depth image of `generate_depth` and the same depth before its uint8 conversion: an HxW float32
        array normalized to [0, 1], from the output of one pipeline call.

        `predicted_depth` is the pipeline's depth already interpolated to the input size, so only the min-max
        min-max normalization that its postprocess applies before quantizing is repeated, computed in float32: a half
        precision model would otherwise leave only a few hundred distinct levels. It is not finite when the depth is
        constant (0/0) or the model overflowed; the caller decides what to do then.
        """
        output = self._pipeline(image)
        depth_map = output["depth"]
        assert isinstance(depth_map, Image.Image)
        depth = output["predicted_depth"].detach().cpu().float().numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            unit = (depth - depth.min()) / (depth.max() - depth.min())
        return depth_map, unit.astype(np.float32)

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        if device is not None and device.type not in {"cpu", "cuda", "xpu"}:
            device = None
        self._pipeline.model.to(device=device, dtype=dtype)
        self._pipeline.device = self._pipeline.model.device

    def calc_size(self) -> int:
        from invokeai.backend.model_manager.load.model_util import calc_module_size

        return calc_module_size(self._pipeline.model)

    @classmethod
    def load_model(cls, model_path: pathlib.Path):
        """Load the model from the given path and return a DepthAnythingPipeline instance."""

        depth_anything_pipeline = pipeline(model=str(model_path), task="depth-estimation", local_files_only=True)
        assert isinstance(depth_anything_pipeline, DepthEstimationPipeline)
        return cls(depth_anything_pipeline)
