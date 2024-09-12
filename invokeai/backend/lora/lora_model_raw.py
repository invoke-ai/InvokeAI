# Copyright (c) 2024 The InvokeAI Development team
"""LoRA model support."""

import bisect
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from safetensors.torch import load_file
from typing_extensions import Self

from invokeai.backend.lora.layers.any_lora_layer import AnyLoRALayer
from invokeai.backend.lora.layers.full_layer import FullLayer
from invokeai.backend.lora.layers.ia3_layer import IA3Layer
from invokeai.backend.lora.layers.loha_layer import LoHALayer
from invokeai.backend.lora.layers.lokr_layer import LoKRLayer
from invokeai.backend.lora.layers.lora_layer import LoRALayer
from invokeai.backend.lora.layers.norm_layer import NormLayer
from invokeai.backend.model_manager import BaseModelType
from invokeai.backend.raw_model import RawModel


class LoRAModelRaw(RawModel):  # (torch.nn.Module):
    _name: str
    layers: Dict[str, AnyLoRALayer]

    def __init__(
        self,
        name: str,
        layers: Dict[str, AnyLoRALayer],
    ):
        self._name = name
        self.layers = layers

    @property
    def name(self) -> str:
        return self._name

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None:
        # TODO: try revert if exception?
        for _key, layer in self.layers.items():
            layer.to(device=device, dtype=dtype)

    def calc_size(self) -> int:
        model_size = 0
        for _, layer in self.layers.items():
            model_size += layer.calc_size()
        return model_size

    @classmethod
    def _convert_sdxl_keys_to_diffusers_format(cls, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Convert the keys of an SDXL LoRA state_dict to diffusers format.

        The input state_dict can be in either Stability AI format or diffusers format. If the state_dict is already in
        diffusers format, then this function will have no effect.

        This function is adapted from:
        https://github.com/bmaltais/kohya_ss/blob/2accb1305979ba62f5077a23aabac23b4c37e935/networks/lora_diffusers.py#L385-L409

        Args:
            state_dict (Dict[str, Tensor]): The SDXL LoRA state_dict.

        Raises:
            ValueError: If state_dict contains an unrecognized key, or not all keys could be converted.

        Returns:
            Dict[str, Tensor]: The diffusers-format state_dict.
        """
        converted_count = 0  # The number of Stability AI keys converted to diffusers format.
        not_converted_count = 0  # The number of keys that were not converted.

        # Get a sorted list of Stability AI UNet keys so that we can efficiently search for keys with matching prefixes.
        # For example, we want to efficiently find `input_blocks_4_1` in the list when searching for
        # `input_blocks_4_1_proj_in`.
        stability_unet_keys = list(SDXL_UNET_STABILITY_TO_DIFFUSERS_MAP)
        stability_unet_keys.sort()

        new_state_dict = {}
        for full_key, value in state_dict.items():
            if full_key.startswith("lora_unet_"):
                search_key = full_key.replace("lora_unet_", "")
                # Use bisect to find the key in stability_unet_keys that *may* match the search_key's prefix.
                position = bisect.bisect_right(stability_unet_keys, search_key)
                map_key = stability_unet_keys[position - 1]
                # Now, check if the map_key *actually* matches the search_key.
                if search_key.startswith(map_key):
                    new_key = full_key.replace(map_key, SDXL_UNET_STABILITY_TO_DIFFUSERS_MAP[map_key])
                    new_state_dict[new_key] = value
                    converted_count += 1
                else:
                    new_state_dict[full_key] = value
                    not_converted_count += 1
            elif full_key.startswith("lora_te1_") or full_key.startswith("lora_te2_"):
                # The CLIP text encoders have the same keys in both Stability AI and diffusers formats.
                new_state_dict[full_key] = value
                continue
            else:
                raise ValueError(f"Unrecognized SDXL LoRA key prefix: '{full_key}'.")

        if converted_count > 0 and not_converted_count > 0:
            raise ValueError(
                f"The SDXL LoRA could only be partially converted to diffusers format. converted={converted_count},"
                f" not_converted={not_converted_count}"
            )

        return new_state_dict

    @classmethod
    def from_checkpoint(
        cls,
        file_path: Union[str, Path],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        base_model: Optional[BaseModelType] = None,
    ) -> Self:
        device = device or torch.device("cpu")
        dtype = dtype or torch.float32

        if isinstance(file_path, str):
            file_path = Path(file_path)

        model = cls(
            name=file_path.stem,
            layers={},
        )

        if file_path.suffix == ".safetensors":
            sd = load_file(file_path.absolute().as_posix(), device="cpu")
        else:
            sd = torch.load(file_path, map_location="cpu")

        state_dict = cls._group_state(sd)

        if base_model == BaseModelType.StableDiffusionXL:
            state_dict = cls._convert_sdxl_keys_to_diffusers_format(state_dict)

        for layer_key, values in state_dict.items():
            # Detect layers according to LyCORIS detection logic(`weight_list_det`)
            # https://github.com/KohakuBlueleaf/LyCORIS/tree/8ad8000efb79e2b879054da8c9356e6143591bad/lycoris/modules

            # lora and locon
            if "lora_up.weight" in values:
                layer: AnyLoRALayer = LoRALayer(layer_key, values)

            # loha
            elif "hada_w1_a" in values:
                layer = LoHALayer(layer_key, values)

            # lokr
            elif "lokr_w1" in values or "lokr_w1_a" in values:
                layer = LoKRLayer(layer_key, values)

            # diff
            elif "diff" in values:
                layer = FullLayer(layer_key, values)

            # ia3
            elif "on_input" in values:
                layer = IA3Layer(layer_key, values)

            # norms
            elif "w_norm" in values:
                layer = NormLayer(layer_key, values)

            else:
                print(f">> Encountered unknown lora layer module in {model.name}: {layer_key} - {list(values.keys())}")
                raise Exception("Unknown lora format!")

            # lower memory consumption by removing already parsed layer values
            state_dict[layer_key].clear()

            layer.to(device=device, dtype=dtype)
            model.layers[layer_key] = layer

        return model

    @staticmethod
    def _group_state(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        state_dict_groupped: Dict[str, Dict[str, torch.Tensor]] = {}

        for key, value in state_dict.items():
            stem, leaf = key.split(".", 1)
            if stem not in state_dict_groupped:
                state_dict_groupped[stem] = {}
            state_dict_groupped[stem][leaf] = value

        return state_dict_groupped


# code from
# https://github.com/bmaltais/kohya_ss/blob/2accb1305979ba62f5077a23aabac23b4c37e935/networks/lora_diffusers.py#L15C1-L97C32
def make_sdxl_unet_conversion_map() -> List[Tuple[str, str]]:
    """Create a dict mapping state_dict keys from Stability AI SDXL format to diffusers SDXL format."""
    unet_conversion_map_layer = []

    for i in range(3):  # num_blocks is 3 in sdxl
        # loop over downblocks/upblocks
        for j in range(2):
            # loop over resnets/attentions for downblocks
            hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
            sd_down_res_prefix = f"input_blocks.{3*i + j + 1}.0."
            unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))

            if i < 3:
                # no attention layers in down_blocks.3
                hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
                sd_down_atn_prefix = f"input_blocks.{3*i + j + 1}.1."
                unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))

        for j in range(3):
            # loop over resnets/attentions for upblocks
            hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
            sd_up_res_prefix = f"output_blocks.{3*i + j}.0."
            unet_conversion_map_layer.append((sd_up_res_prefix, hf_up_res_prefix))

            # if i > 0: commentout for sdxl
            # no attention layers in up_blocks.0
            hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
            sd_up_atn_prefix = f"output_blocks.{3*i + j}.1."
            unet_conversion_map_layer.append((sd_up_atn_prefix, hf_up_atn_prefix))

        if i < 3:
            # no downsample in down_blocks.3
            hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
            sd_downsample_prefix = f"input_blocks.{3*(i+1)}.0.op."
            unet_conversion_map_layer.append((sd_downsample_prefix, hf_downsample_prefix))

            # no upsample in up_blocks.3
            hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
            sd_upsample_prefix = f"output_blocks.{3*i + 2}.{2}."  # change for sdxl
            unet_conversion_map_layer.append((sd_upsample_prefix, hf_upsample_prefix))

    hf_mid_atn_prefix = "mid_block.attentions.0."
    sd_mid_atn_prefix = "middle_block.1."
    unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))

    for j in range(2):
        hf_mid_res_prefix = f"mid_block.resnets.{j}."
        sd_mid_res_prefix = f"middle_block.{2*j}."
        unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))

    unet_conversion_map_resnet = [
        # (stable-diffusion, HF Diffusers)
        ("in_layers.0.", "norm1."),
        ("in_layers.2.", "conv1."),
        ("out_layers.0.", "norm2."),
        ("out_layers.3.", "conv2."),
        ("emb_layers.1.", "time_emb_proj."),
        ("skip_connection.", "conv_shortcut."),
    ]

    unet_conversion_map = []
    for sd, hf in unet_conversion_map_layer:
        if "resnets" in hf:
            for sd_res, hf_res in unet_conversion_map_resnet:
                unet_conversion_map.append((sd + sd_res, hf + hf_res))
        else:
            unet_conversion_map.append((sd, hf))

    for j in range(2):
        hf_time_embed_prefix = f"time_embedding.linear_{j+1}."
        sd_time_embed_prefix = f"time_embed.{j*2}."
        unet_conversion_map.append((sd_time_embed_prefix, hf_time_embed_prefix))

    for j in range(2):
        hf_label_embed_prefix = f"add_embedding.linear_{j+1}."
        sd_label_embed_prefix = f"label_emb.0.{j*2}."
        unet_conversion_map.append((sd_label_embed_prefix, hf_label_embed_prefix))

    unet_conversion_map.append(("input_blocks.0.0.", "conv_in."))
    unet_conversion_map.append(("out.0.", "conv_norm_out."))
    unet_conversion_map.append(("out.2.", "conv_out."))

    return unet_conversion_map


SDXL_UNET_STABILITY_TO_DIFFUSERS_MAP = {
    sd.rstrip(".").replace(".", "_"): hf.rstrip(".").replace(".", "_") for sd, hf in make_sdxl_unet_conversion_map()
}
