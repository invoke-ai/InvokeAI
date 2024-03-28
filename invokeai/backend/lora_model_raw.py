from pathlib import Path
from typing import Optional, Union

import torch
from diffusers.loaders.lora import LoraLoaderMixin
from typing_extensions import Self


class LoRAModelRaw:
    def __init__(
        self,
        name: str,
        state_dict: dict[str, torch.Tensor],
        network_alphas: Optional[dict[str, float]],
    ):
        self._name = name
        self.state_dict = state_dict
        self.network_alphas = network_alphas

    @property
    def name(self) -> str:
        return self._name

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None:
        for key, layer in self.state_dict.items():
            self.state_dict[key] = layer.to(device=device, dtype=dtype)

    def calc_size(self) -> int:
        """Calculate the size of the model in bytes."""
        model_size = 0
        for layer in self.state_dict.values():
            model_size += layer.numel() * layer.element_size()
        return model_size

    @classmethod
    def from_checkpoint(
        cls, file_path: Union[str, Path], device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ) -> Self:
        """This function is based on diffusers LoraLoaderMixin.load_lora_weights()."""

        file_path = Path(file_path)
        if file_path.is_dir():
            raise NotImplementedError("LoRA models from directories are not yet supported.")

        dir_path = file_path.parent
        file_name = file_path.name

        state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(
            pretrained_model_name_or_path_or_dict=str(file_path), local_files_only=True, weight_name=str(file_name)
        )

        is_correct_format = all("lora" in key for key in state_dict.keys())
        if not is_correct_format:
            raise ValueError("Invalid LoRA checkpoint.")

        model = cls(
            # TODO(ryand): Handle both files and directories here?
            name=Path(file_path).stem,
            state_dict=state_dict,
            network_alphas=network_alphas,
        )

        device = device or torch.device("cpu")
        dtype = dtype or torch.float32
        model.to(device=device, dtype=dtype)
        return model
