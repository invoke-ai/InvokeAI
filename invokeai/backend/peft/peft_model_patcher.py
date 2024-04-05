from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Tuple

import torch
from diffusers.models.lora import text_encoder_attn_modules, text_encoder_mlp_modules
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.utils.peft_utils import get_peft_kwargs, scale_lora_layers
from diffusers.utils.state_dict_utils import convert_state_dict_to_peft, convert_unet_state_dict_to_peft
from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict

from invokeai.backend.peft.peft_model import PeftModel

UNET_NAME = "unet"


class PeftModelPatcher:
    @classmethod
    @contextmanager
    @torch.no_grad()
    def apply_peft_model_to_text_encoder(
        cls,
        text_encoder: torch.nn.Module,
        peft_models: Iterator[Tuple[PeftModel, float]],
        prefix: str,
    ):
        original_weights = {}

        try:
            for peft_model, peft_model_weight in peft_models:
                keys = list(peft_model.state_dict.keys())

                # Load the layers corresponding to text encoder and make necessary adjustments.
                text_encoder_keys = [k for k in keys if k.startswith(prefix) and k.split(".")[0] == prefix]
                text_encoder_lora_state_dict = {
                    k.replace(f"{prefix}.", ""): v for k, v in peft_model.state_dict.items() if k in text_encoder_keys
                }

                if len(text_encoder_lora_state_dict) == 0:
                    continue

                if peft_model.name in getattr(text_encoder, "peft_config", {}):
                    raise ValueError(f"Adapter name {peft_model.name} already in use in the text encoder ({prefix}).")

                rank = {}
                # TODO(ryand): Is this necessary?
                # text_encoder_lora_state_dict = convert_state_dict_to_diffusers(text_encoder_lora_state_dict)

                text_encoder_lora_state_dict = convert_state_dict_to_peft(text_encoder_lora_state_dict)

                for name, _ in text_encoder_attn_modules(text_encoder):
                    rank_key = f"{name}.out_proj.lora_B.weight"
                    rank[rank_key] = text_encoder_lora_state_dict[rank_key].shape[1]

                patch_mlp = any(".mlp." in key for key in text_encoder_lora_state_dict.keys())
                if patch_mlp:
                    for name, _ in text_encoder_mlp_modules(text_encoder):
                        rank_key_fc1 = f"{name}.fc1.lora_B.weight"
                        rank_key_fc2 = f"{name}.fc2.lora_B.weight"

                        rank[rank_key_fc1] = text_encoder_lora_state_dict[rank_key_fc1].shape[1]
                        rank[rank_key_fc2] = text_encoder_lora_state_dict[rank_key_fc2].shape[1]

                network_alphas = peft_model.network_alphas
                if network_alphas is not None:
                    alpha_keys = [
                        k for k in network_alphas.keys() if k.startswith(prefix) and k.split(".")[0] == prefix
                    ]
                    network_alphas = {
                        k.replace(f"{prefix}.", ""): v for k, v in network_alphas.items() if k in alpha_keys
                    }

                lora_config_kwargs = get_peft_kwargs(rank, network_alphas, text_encoder_lora_state_dict, is_unet=False)
                lora_config_kwargs["inference_mode"] = True
                lora_config = LoraConfig(**lora_config_kwargs)

                new_text_encoder = inject_adapter_in_model(lora_config, text_encoder, peft_model.name)
                incompatible_keys = set_peft_model_state_dict(
                    new_text_encoder, text_encoder_lora_state_dict, peft_model.name
                )
                if incompatible_keys is not None:
                    # check only for unexpected keys
                    unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                    if unexpected_keys:
                        raise ValueError(f"Failed to inject unexpected PEFT keys: {unexpected_keys}")

                # inject LoRA layers and load the state dict
                # in transformers we automatically check whether the adapter name is already in use or not
                # text_encoder.load_adapter(
                #     adapter_name=adapter_name,
                #     adapter_state_dict=text_encoder_lora_state_dict,
                #     peft_config=lora_config,
                # )

                scale_lora_layers(text_encoder, weight=peft_model_weight)
                text_encoder.to(device=text_encoder.device, dtype=text_encoder.dtype)

            yield
        finally:
            # TODO
            pass
            # for module_key, weight in original_weights.items():
            #     model.get_submodule(module_key).weight.copy_(weight)

    @classmethod
    @contextmanager
    @torch.no_grad()
    def apply_peft_model_to_unet(
        cls,
        unet: UNet2DConditionModel,
        peft_models: Iterator[Tuple[PeftModel, float]],
    ):
        try:
            for peft_model, peft_model_weight in peft_models:
                keys = list(peft_model.state_dict.keys())

                unet_keys = [k for k in keys if k.startswith(UNET_NAME)]
                state_dict = {
                    k.replace(f"{UNET_NAME}.", ""): v for k, v in peft_model.state_dict.items() if k in unet_keys
                }

                network_alphas = peft_model.network_alphas
                if network_alphas is not None:
                    alpha_keys = [k for k in network_alphas.keys() if k.startswith(UNET_NAME)]
                    network_alphas = {
                        k.replace(f"{UNET_NAME}.", ""): v for k, v in network_alphas.items() if k in alpha_keys
                    }

                if len(state_dict) == 0:
                    continue

                if peft_model.name in getattr(unet, "peft_config", {}):
                    raise ValueError(f"Adapter name {peft_model.name} already in use in the Unet.")

                state_dict = convert_unet_state_dict_to_peft(state_dict)

                if network_alphas is not None:
                    # The alphas state dict have the same structure as Unet, thus we convert it to peft format using
                    # `convert_unet_state_dict_to_peft` method.
                    network_alphas = convert_unet_state_dict_to_peft(network_alphas)

                rank = {}
                for key, val in state_dict.items():
                    if "lora_B" in key:
                        rank[key] = val.shape[1]

                lora_config_kwargs = get_peft_kwargs(rank, network_alphas, state_dict, is_unet=True)
                lora_config_kwargs["inference_mode"] = True
                lora_config = LoraConfig(**lora_config_kwargs)

                inject_adapter_in_model(lora_config, unet, adapter_name=peft_model.name)
                incompatible_keys = set_peft_model_state_dict(unet, state_dict, peft_model.name)
                if incompatible_keys is not None:
                    # check only for unexpected keys
                    unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                    if unexpected_keys:
                        raise ValueError(f"Failed to inject unexpected PEFT keys: {unexpected_keys}")

                # TODO(ryand): What does this do?
                unet.load_attn_procs(state_dict, network_alphas=network_alphas, low_cpu_mem_usage=True)

                # TODO(ryand): Apply the lora weight. Where does diffusers do this? They don't seem to do it when they
                # patch the UNet.
            yield
        finally:
            # TODO
            pass
            # for module_key, weight in original_weights.items():
            #     model.get_submodule(module_key).weight.copy_(weight)

    @classmethod
    @contextmanager
    @torch.no_grad()
    def apply_peft_patch(
        cls,
        model: torch.nn.Module,
        peft_models: Iterator[Tuple[PeftModel, float]],
        prefix: str,
    ):
        original_weights = {}

        model_state_dict = model.state_dict()
        try:
            for peft_model, peft_model_weight in peft_models:
                for layer_key, layer in peft_model.state_dict.items():
                    if not layer_key.startswith(prefix):
                        continue

                    module_key = layer_key.replace(prefix + ".", "")
                    # TODO(ryand): Make this work.

                    module = model_state_dict[module_key]

                    # All of the LoRA weight calculations will be done on the same device as the module weight.
                    # (Performance will be best if this is a CUDA device.)
                    device = module.weight.device
                    dtype = module.weight.dtype

                    if module_key not in original_weights:
                        # TODO(ryand): Set non_blocking = True?
                        original_weights[module_key] = module.weight.detach().to(device="cpu", copy=True)

                    layer_scale = layer.alpha / layer.rank if (layer.alpha and layer.rank) else 1.0

                    # We intentionally move to the target device first, then cast. Experimentally, this was found to
                    # be significantly faster for 16-bit CPU tensors being moved to a CUDA device than doing the
                    # same thing in a single call to '.to(...)'.
                    layer.to(device=device)
                    layer.to(dtype=torch.float32)
                    # TODO(ryand): Using torch.autocast(...) over explicit casting may offer a speed benefit on CUDA
                    # devices here. Experimentally, it was found to be very slow on CPU. More investigation needed.
                    layer_weight = layer.get_weight(module.weight) * (lora_weight * layer_scale)
                    layer.to(device=torch.device("cpu"))

                    assert isinstance(layer_weight, torch.Tensor)  # mypy thinks layer_weight is a float|Any ??!
                    if module.weight.shape != layer_weight.shape:
                        # TODO: debug on lycoris
                        assert hasattr(layer_weight, "reshape")
                        layer_weight = layer_weight.reshape(module.weight.shape)

                    assert isinstance(layer_weight, torch.Tensor)  # mypy thinks layer_weight is a float|Any ??!
                    module.weight += layer_weight.to(dtype=dtype)
            yield
        finally:
            for module_key, weight in original_weights.items():
                model.get_submodule(module_key).weight.copy_(weight)
