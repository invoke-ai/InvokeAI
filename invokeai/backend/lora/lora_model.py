from pathlib import Path
from typing import Optional, Union

import torch

from invokeai.backend.lora.full_layer import FullLayer
from invokeai.backend.lora.ia3_layer import IA3Layer
from invokeai.backend.lora.loha_layer import LoHALayer
from invokeai.backend.lora.lokr_layer import LoKRLayer
from invokeai.backend.lora.lora_layer import LoRALayer
from invokeai.backend.lora.sdxl_state_dict_utils import convert_sdxl_keys_to_diffusers_format
from invokeai.backend.model_manager import BaseModelType
from invokeai.backend.util.serialization import load_state_dict

AnyLoRALayer = Union[LoRALayer, LoHALayer, LoKRLayer, FullLayer, IA3Layer]


class LoRAModelRaw(torch.nn.Module):
    def __init__(
        self,
        name: str,
        layers: dict[str, AnyLoRALayer],
    ):
        super().__init__()
        self._name = name
        self.layers = layers

    @property
    def name(self) -> str:
        return self._name

    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        # TODO: try revert if exception?
        for _key, layer in self.layers.items():
            layer.to(device=device, dtype=dtype)

    def calc_size(self) -> int:
        model_size = 0
        for _, layer in self.layers.items():
            model_size += layer.calc_size()
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

        model_name = file_path.stem

        sd = load_state_dict(file_path, device=str(device))
        state_dict = cls._group_state(sd)

        if base_model == BaseModelType.StableDiffusionXL:
            state_dict = convert_sdxl_keys_to_diffusers_format(state_dict)

        layers: dict[str, AnyLoRALayer] = {}
        for layer_key, values in state_dict.items():
            # lora and locon
            if "lora_down.weight" in values:
                layer: AnyLoRALayer = LoRALayer(layer_key, values)

            # loha
            elif "hada_w1_b" in values:
                layer = LoHALayer(layer_key, values)

            # lokr
            elif "lokr_w1_b" in values or "lokr_w1" in values:
                layer = LoKRLayer(layer_key, values)

            # diff
            elif "diff" in values:
                layer = FullLayer(layer_key, values)

            # ia3
            elif "weight" in values and "on_input" in values:
                layer = IA3Layer(layer_key, values)

            else:
                raise ValueError(f"Unknown lora layer module in {model_name}: {layer_key}: {list(values.keys())}")

            # lower memory consumption by removing already parsed layer values
            state_dict[layer_key].clear()

            layer.to(device=device, dtype=dtype)
            layers[layer_key] = layer

        return cls(name=model_name, layers=layers)

    @staticmethod
    def _group_state(state_dict: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
        state_dict_groupped: dict[str, dict[str, torch.Tensor]] = {}

        for key, value in state_dict.items():
            stem, leaf = key.split(".", 1)
            if stem not in state_dict_groupped:
                state_dict_groupped[stem] = {}
            state_dict_groupped[stem][leaf] = value

        return state_dict_groupped
