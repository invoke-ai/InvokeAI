from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from spandrel import ImageModelDescriptor, ModelLoader

from invokeai.backend.raw_model import RawModel


class SpandrelImageToImageModel(RawModel):
    """A wrapper for a Spandrel Image-to-Image model.

    The main reason for having a wrapper class is to integrate with the type handling of RawModel.
    """

    def __init__(self, spandrel_model: ImageModelDescriptor[Any]):
        self._spandrel_model = spandrel_model

    @staticmethod
    def pil_to_tensor(image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to the torch.Tensor format expected by SpandrelImageToImageModel.run().

        Args:
            image (Image.Image): A PIL Image with shape (H, W, C) and values in the range [0, 255].

        Returns:
            torch.Tensor: A torch.Tensor with shape (N, C, H, W) and values in the range [0, 1].
        """
        image_np = np.array(image)
        # (H, W, C) -> (C, H, W)
        image_np = np.transpose(image_np, (2, 0, 1))
        image_np = image_np / 255
        image_tensor = torch.from_numpy(image_np).float()
        # (C, H, W) -> (N, C, H, W)
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor

    @staticmethod
    def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        """Convert a torch.Tensor produced by SpandrelImageToImageModel.run() to a PIL Image.

        Args:
            tensor (torch.Tensor): A torch.Tensor with shape (N, C, H, W) and values in the range [0, 1].

        Returns:
            Image.Image: A PIL Image with shape (H, W, C) and values in the range [0, 255].
        """
        # (N, C, H, W) -> (C, H, W)
        tensor = tensor.squeeze(0)
        # (C, H, W) -> (H, W, C)
        tensor = tensor.permute(1, 2, 0)
        tensor = tensor.clamp(0, 1)
        tensor = (tensor * 255).cpu().detach().numpy().astype(np.uint8)
        image = Image.fromarray(tensor)
        return image

    def run(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Run the image-to-image model.

        Args:
            image_tensor (torch.Tensor): A torch.Tensor with shape (N, C, H, W) and values in the range [0, 1].
        """
        return self._spandrel_model(image_tensor)

    @classmethod
    def load_from_file(cls, file_path: str | Path):
        model = ModelLoader().load_from_file(file_path)
        if not isinstance(model, ImageModelDescriptor):
            raise ValueError(
                f"Loaded a spandrel model of type '{type(model)}'. Only image-to-image models are supported "
                "('ImageModelDescriptor')."
            )

        return cls(spandrel_model=model)

    @classmethod
    def load_from_state_dict(cls, state_dict: dict[str, torch.Tensor]):
        model = ModelLoader().load_from_state_dict(state_dict)
        if not isinstance(model, ImageModelDescriptor):
            raise ValueError(
                f"Loaded a spandrel model of type '{type(model)}'. Only image-to-image models are supported "
                "('ImageModelDescriptor')."
            )

        return cls(spandrel_model=model)

    def supports_dtype(self, dtype: torch.dtype) -> bool:
        """Check if the model supports the given dtype."""
        if dtype == torch.float16:
            return self._spandrel_model.supports_half
        elif dtype == torch.bfloat16:
            return self._spandrel_model.supports_bfloat16
        elif dtype == torch.float32:
            # All models support float32.
            return True
        else:
            raise ValueError(f"Unexpected dtype '{dtype}'.")

    def get_model_type_name(self) -> str:
        """The model type name. Intended for logging / debugging purposes. Do not rely on this field remaining
        consistent over time.
        """
        return str(type(self._spandrel_model.model))

    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = False,
    ) -> None:
        """Note: Some models have limited dtype support. Call supports_dtype(...) to check if the dtype is supported.
        Note: The non_blocking parameter is currently ignored."""
        # TODO(ryand): spandrel.ImageModelDescriptor.to(...) does not support non_blocking. We will have to access the
        # model directly if we want to apply this optimization.
        self._spandrel_model.to(device=device, dtype=dtype)

    @property
    def device(self) -> torch.device:
        """The device of the underlying model."""
        return self._spandrel_model.device

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the underlying model."""
        return self._spandrel_model.dtype

    @property
    def scale(self) -> int:
        """The scale of the model (e.g. 1x, 2x, 4x, etc.)."""
        return self._spandrel_model.scale

    def calc_size(self) -> int:
        """Get size of the model in memory in bytes."""
        # HACK(ryand): Fix this issue with circular imports.
        from invokeai.backend.model_manager.load.model_util import calc_module_size

        return calc_module_size(self._spandrel_model.model)
