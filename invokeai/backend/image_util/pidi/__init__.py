# Adapted from https://github.com/huggingface/controlnet_aux

import pathlib

import cv2
import huggingface_hub
import numpy as np
import torch
from einops import rearrange
from PIL import Image

from invokeai.backend.image_util.pidi.model import PiDiNet, pidinet
from invokeai.backend.image_util.util import nms, normalize_image_channel_count, np_to_pil, pil_to_np, safe_step


class PIDINetDetector:
    """Simple wrapper around a PiDiNet model for edge detection."""

    hf_repo_id = "lllyasviel/Annotators"
    hf_filename = "table5_pidinet.pth"

    @classmethod
    def get_model_url(cls) -> str:
        """Get the URL to download the model from the Hugging Face Hub."""
        return huggingface_hub.hf_hub_url(cls.hf_repo_id, cls.hf_filename)

    @classmethod
    def load_model(cls, model_path: pathlib.Path) -> PiDiNet:
        """Load the model from a file."""

        model = pidinet()
        model.load_state_dict({k.replace("module.", ""): v for k, v in torch.load(model_path)["state_dict"].items()})
        model.eval()
        return model

    def __init__(self, model: PiDiNet) -> None:
        self.model = model

    def to(self, device: torch.device):
        self.model.to(device)
        return self

    def run(
        self, image: Image.Image, quantize_edges: bool = False, scribble: bool = False, apply_filter: bool = False
    ) -> Image.Image:
        """Processes an image and returns the detected edges."""

        device = next(iter(self.model.parameters())).device

        np_img = pil_to_np(image)
        np_img = normalize_image_channel_count(np_img)

        assert np_img.ndim == 3

        bgr_img = np_img[:, :, ::-1].copy()

        with torch.no_grad():
            image_pidi = torch.from_numpy(bgr_img).float().to(device)
            image_pidi = image_pidi / 255.0
            image_pidi = rearrange(image_pidi, "h w c -> 1 c h w")
            edge = self.model(image_pidi)[-1]
            edge = edge.cpu().numpy()
            if apply_filter:
                edge = edge > 0.5
            if quantize_edges:
                edge = safe_step(edge)
            edge = (edge * 255.0).clip(0, 255).astype(np.uint8)

        detected_map = edge[0, 0]

        if scribble:
            detected_map = nms(detected_map, 127, 3.0)
            detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
            detected_map[detected_map > 4] = 255
            detected_map[detected_map < 255] = 0

        output_img = np_to_pil(detected_map)

        return output_img
