from pathlib import Path
from typing import Any, Optional

import torch
from spandrel import ImageModelDescriptor, ModelLoader

from invokeai.backend.raw_model import RawModel


class SpandrelImageToImageModel(RawModel):
    """A wrapper for a Spandrel Image-to-Image model.

    The main reason for having a wrapper class is to integrate with the type handling of RawModel.
    """

    def __init__(self, spandrel_model: ImageModelDescriptor[Any]):
        self._spandrel_model = spandrel_model

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
        # TODO(ryand): spandrel.ImageModelDescriptor.to(...) does not support non_blocking. We will access the model
        # directly if we want to apply this optimization.
        self._spandrel_model.to(device=device, dtype=dtype)
