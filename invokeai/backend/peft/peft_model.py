from pathlib import Path
from typing import Optional, Union

import torch

from invokeai.backend.model_manager.config import BaseModelType
from invokeai.backend.peft.sdxl_format_utils import convert_sdxl_keys_to_diffusers_format
from invokeai.backend.util.serialization import load_state_dict


class PeftModel:
    """A class for loading and managing parameter-efficient fine-tuning models."""

    def __init__(
        self,
        name: str,
        state_dict: dict[str, torch.Tensor],
        network_alphas: dict[str, torch.Tensor],
    ):
        self.name = name
        self.state_dict = state_dict
        self.network_alphas = network_alphas

    def calc_size(self) -> int:
        model_size = 0
        for tensor in self.state_dict.values():
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

        state_dict = load_state_dict(file_path, device=str(device))
        if base_model == BaseModelType.StableDiffusionXL:
            state_dict = convert_sdxl_keys_to_diffusers_format(state_dict)

        # TODO(ryand): We shouldn't be using an unexported function from diffusers here. Consider opening an upstream PR
        # to move this function to state_dict_utils.py.
        # state_dict, network_alphas = _convert_kohya_lora_to_diffusers(state_dict)
        return cls(name=file_path.stem, state_dict=state_dict, network_alphas=network_alphas)
