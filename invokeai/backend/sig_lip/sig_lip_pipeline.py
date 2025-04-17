import torch
from PIL import Image
from transformers import SiglipImageProcessor, SiglipVisionModel


class SigLipPipeline:
    """A wrapper for a SigLIP model + processor."""

    def __init__(
        self,
        siglip_processor: SiglipImageProcessor,
        siglip_model: SiglipVisionModel,
    ):
        self._siglip_processor = siglip_processor
        self._siglip_model = siglip_model

    def encode_image(self, x: Image.Image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        imgs = self._siglip_processor.preprocess(images=[x], do_resize=True, return_tensors="pt", do_convert_rgb=True)
        encoded_x = self._siglip_model(**imgs.to(device=device, dtype=dtype)).last_hidden_state
        return encoded_x
