from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import SiglipImageProcessor, SiglipVisionModel

from invokeai.backend.raw_model import RawModel


class SigLipPipeline(RawModel):
    """A wrapper for a SigLIP model + processor."""

    def __init__(
        self,
        siglip_processor: SiglipImageProcessor,
        siglip_model: SiglipVisionModel,
    ):
        self._siglip_processor = siglip_processor
        self._siglip_model = siglip_model

    @classmethod
    def load_from_path(cls, path: str | Path):
        siglip_model = SiglipVisionModel.from_pretrained(path, local_files_only=True)
        assert isinstance(siglip_model, SiglipVisionModel)
        siglip_processor = SiglipImageProcessor.from_pretrained(path, local_files_only=True)
        assert isinstance(siglip_processor, SiglipImageProcessor)
        return cls(siglip_processor, siglip_model)

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None:
        self._siglip_model.to(device=device, dtype=dtype)

    def encode_image(self, x: Image.Image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        imgs = self._siglip_processor.preprocess(images=[x], do_resize=True, return_tensors="pt", do_convert_rgb=True)
        encoded_x = self._siglip_model(**imgs.to(device=device, dtype=dtype)).last_hidden_state
        return encoded_x

    def calc_size(self) -> int:
        """Get size of the model in memory in bytes."""
        # HACK(ryand): Fix this issue with circular imports.
        from invokeai.backend.model_manager.load.model_util import calc_module_size

        return calc_module_size(self._siglip_model)
