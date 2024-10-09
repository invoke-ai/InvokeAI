# Adapted from https://github.com/huggingface/controlnet_aux

import pathlib

import cv2
import huggingface_hub
import numpy as np
import torch
from PIL import Image

from invokeai.backend.image_util.mlsd.models.mbv2_mlsd_large import MobileV2_MLSD_Large
from invokeai.backend.image_util.mlsd.utils import pred_lines
from invokeai.backend.image_util.util import np_to_pil, pil_to_np, resize_to_multiple


class MLSDDetector:
    """Simple wrapper around a MLSD model for detecting edges as line segments in an image."""

    hf_repo_id = "lllyasviel/ControlNet"
    hf_filename = "annotator/ckpts/mlsd_large_512_fp32.pth"

    @classmethod
    def get_model_url(cls) -> str:
        """Get the URL to download the model from the Hugging Face Hub."""

        return huggingface_hub.hf_hub_url(cls.hf_repo_id, cls.hf_filename)

    @classmethod
    def load_model(cls, model_path: pathlib.Path) -> MobileV2_MLSD_Large:
        """Load the model from a file."""

        model = MobileV2_MLSD_Large()
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        return model

    def __init__(self, model: MobileV2_MLSD_Large) -> None:
        self.model = model

    def to(self, device: torch.device):
        self.model.to(device)
        return self

    def run(self, image: Image.Image, score_threshold: float = 0.1, distance_threshold: float = 20.0) -> Image.Image:
        """Processes an image and returns the detected edges."""

        np_img = pil_to_np(image)

        height, width, _channels = np_img.shape

        # This model requires the input image to have a resolution that is a multiple of 64
        np_img = resize_to_multiple(np_img, 64)
        img_output = np.zeros_like(np_img)

        with torch.no_grad():
            lines = pred_lines(np_img, self.model, [np_img.shape[0], np_img.shape[1]], score_threshold, distance_threshold)
            for line in lines:
                x_start, y_start, x_end, y_end = [int(val) for val in line]
                cv2.line(img_output, (x_start, y_start), (x_end, y_end), [255, 255, 255], 1)

        detected_map = img_output[:, :, 0]

        # Back to the original size
        output_image = cv2.resize(detected_map, (width, height), interpolation=cv2.INTER_LINEAR)

        return np_to_pil(output_image)
