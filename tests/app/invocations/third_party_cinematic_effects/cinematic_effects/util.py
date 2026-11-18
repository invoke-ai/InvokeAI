import pathlib
from typing import Optional

import numpy as np
import torch
from invokeai.backend.raw_model import RawModel
from invokeai.invocation_api import oklch_from_srgb, srgb_from_oklch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


def rgb_to_oklch(rgb: np.ndarray) -> np.ndarray:
    rgb_tensor = torch.from_numpy(np.clip(rgb, 0.0, 1.0)).permute(2, 0, 1).to(dtype=torch.float64)
    return oklch_from_srgb(rgb_tensor).permute(1, 2, 0).numpy()


def oklch_to_rgb(oklch: np.ndarray) -> np.ndarray:
    oklch_tensor = torch.from_numpy(oklch).permute(2, 0, 1).to(dtype=torch.float64)
    return srgb_from_oklch(oklch_tensor).permute(1, 2, 0).numpy()


class MyDepthMapPipeline(RawModel):
    """Custom wrapper for the Depth Estimation pipeline from transformers adding compatibility
    for Invoke's Model Management System plus returning fp depth"""

    def __init__(self, image_processor: AutoImageProcessor, depth_model: AutoModelForDepthEstimation) -> None:
        print("__init__")
        self._image_processor = image_processor
        self._depth_model = depth_model

    def generate_depth(self, image: Image.Image) -> Image.Image:
        print("GD")
        # Prepare image for the model
        inputs = self._image_processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self._depth_model(**inputs)

        # Interpolate to original size
        post_processed_output = self._image_processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(image.height, image.width)],
        )

        # Visualize the prediction
        predicted_depth = post_processed_output[0]["predicted_depth"]
        depth_map = predicted_depth * 255 / predicted_depth.max()
        depth_map = depth_map.detach().cpu().numpy()
        print("DM")

        return depth_map

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        if device is not None and device.type not in {"cpu", "cuda"}:
            device = None
        self._image_processor.model.to(device=device, dtype=dtype)
        self._image_processor.device = self._pipeline.model.device
        self._depth_model.model.to(device=device, dtype=dtype)
        self._depth_model.device = self._pipeline.model.device

    def calc_size(self) -> int:
        from invokeai.backend.model_manager.load.model_util import calc_module_size

        return calc_module_size(self._image_processor.model) + calc_module_size(self._depth_model)

    @classmethod
    def load_model(cls, model_path: pathlib.Path):
        """Load the model from the given path and return a MyDepthMapPipeline instance."""

        print("__load_model__")
        image_processor = AutoImageProcessor.from_pretrained(str(model_path))
        depth_model = AutoModelForDepthEstimation.from_pretrained(str(model_path))
        return MyDepthMapPipeline(image_processor, depth_model)
