import pathlib
from typing import Optional

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
        pipeline_result = self._pipeline(image)
        predicted_depth = pipeline_result["predicted_depth"]
        assert isinstance(predicted_depth, torch.Tensor)

        # Convert to PIL Image.
        # Note: The pipeline already returns a PIL Image (pipeline_result["depth"]), but it contains artifacts as
        # described here: https://github.com/invoke-ai/InvokeAI/issues/7358.
        # We implement custom post-processing logic to avoid the artifacts.
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1), size=image.size[::-1], mode="bilinear", align_corners=False
        )
        prediction = prediction / prediction.max()
        output = prediction.squeeze().cpu().numpy()
        output = (output * 255).clip(0, 255)
        formatted = output.astype("uint8")
        depth = Image.fromarray(formatted)
        return depth

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        if device is not None and device.type not in {"cpu", "cuda"}:
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
