from pathlib import Path
from typing import Optional, Union

import torch
from safetensors.torch import load_file

from invokeai.backend.model_manager.config import BaseModelType


class PeftModel:
    """A class for loading and managing parameter-efficient fine-tuning models."""

    def __init__(
        self,
        name: str,
        state_dict: dict[str, torch.Tensor],
    ):
        self._name = name
        self._state_dict = state_dict

    @property
    def name(self) -> str:
        return self._name

    def calc_size(self) -> int:
        model_size = 0
        for tensor in self._state_dict.values():
            model_size += tensor.nelement() * tensor.element_size()
        return model_size

    @classmethod
    def from_checkpoint(
        cls,
        file_path: Union[str, Path],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        base_model: Optional[BaseModelType] = None,
    ):
        device = device or torch.device("cpu")
        dtype = dtype or torch.float32

        file_path = Path(file_path)

        # TODO(ryand): Implement a helper function for this. This logic is duplicated repeatedly.
        if file_path.suffix == ".safetensors":
            state_dict = load_file(file_path, device="cpu")
        else:
            state_dict = torch.load(file_path, map_location="cpu")

        # TODO(ryand):
        # - Detect state_dict format
        # - Convert state_dict to diffusers format if necessary

        # if base_model == BaseModelType.StableDiffusionXL:
        #     state_dict = cls._convert_sdxl_keys_to_diffusers_format(state_dict)
        return cls(name=file_path.stem, state_dict=state_dict)
