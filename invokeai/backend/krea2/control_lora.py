"""Krea-2 Control LoRA: block LoRAs plus a widened input projection, trained by
https://github.com/Tanmaypatil123/Krea-2-controlnet (e.g. ``Patil/Krea-2-depth-controlnet``).

The checkpoint replaces the transformer's input projection (``img_in``, native ``first``) with a trained
``Linear(2 * 64, hidden)`` that takes ``[noisy latent tokens | control latent tokens]`` on the feature axis, and adds
rank-r LoRAs on every block's attention and MLP Linears. Its reference pipeline (``forward_control``) projects the
target tokens with the checkpoint's whole projection, weight and bias, in both CFG passes; the LoRA scale reaches the
block LoRAs only.

The block LoRAs are an ordinary Krea-2 LoRA for ``LayerPatcher``. The projection goes on as a forward hook on
``img_in`` instead of a patched weight, so it works whatever class ``img_in`` is (plain, autocast, GGUF, fp8, int8 or
sidecar-patched), and it re-projects only the target tokens, leaving joint edit references as ``img_in`` made them.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, Optional

import torch
import torch.nn.functional as F

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.lora_conversions.krea2_lora_constants import KREA2_CONTROL_LORA_PROJECTION_KEYS
from invokeai.backend.patches.lora_conversions.krea2_lora_conversion_utils import lora_model_from_krea2_state_dict
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.util.calc_tensor_size import calc_tensors_size


class Krea2ControlLoRAPatch(ModelPatchRaw):
    """The block LoRA layers, plus the widened input projection the model cache moves and bills with them."""

    def __init__(
        self,
        layers: Mapping[str, BaseLayerPatch],
        input_projection_weight: torch.Tensor,
        input_projection_bias: torch.Tensor,
    ):
        super().__init__(layers)
        self.input_projection_weight = input_projection_weight
        self.input_projection_bias = input_projection_bias

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None:
        super().to(device=device, dtype=dtype)
        self.input_projection_weight = self.input_projection_weight.to(device=device, dtype=dtype)
        self.input_projection_bias = self.input_projection_bias.to(device=device, dtype=dtype)

    def calc_size(self) -> int:
        return super().calc_size() + calc_tensors_size([self.input_projection_weight, self.input_projection_bias])


def krea2_control_lora_from_state_dict(state_dict: dict[str, torch.Tensor]) -> Krea2ControlLoRAPatch:
    """Convert a Krea-2 Control LoRA file. Its ``.A``/``.B`` halves are LoRA down/up with alpha = rank."""
    lora_state_dict: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key in KREA2_CONTROL_LORA_PROJECTION_KEYS:
            continue
        if key.endswith(".A"):
            key = f"{key[:-2]}.lora_A.weight"
        elif key.endswith(".B"):
            key = f"{key[:-2]}.lora_B.weight"
        lora_state_dict[key] = value
    weight_key, bias_key = KREA2_CONTROL_LORA_PROJECTION_KEYS
    return Krea2ControlLoRAPatch(
        layers=lora_model_from_krea2_state_dict(lora_state_dict, alpha=None).layers,
        input_projection_weight=state_dict[weight_key],
        input_projection_bias=state_dict[bias_key],
    )


class Krea2ControlLoRAExtension:
    """Projects one denoise's target tokens through a Control LoRA's widened input projection.

    ``control_tokens`` are the packed control latents, ``(1, target_len, 64)``. The hook re-projects only while
    :meth:`armed`, which the denoise enters around its target predictions alone.
    """

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, control_tokens: torch.Tensor) -> None:
        if control_tokens.dim() != 3 or control_tokens.shape[0] != 1:
            raise ValueError(f"Krea-2 control tokens must be (1, tokens, channels), got {tuple(control_tokens.shape)}.")
        self._weight = weight
        self._bias = bias
        self._control_tokens = control_tokens
        self._armed = False

    @contextmanager
    def install(self, transformer: torch.nn.Module) -> Iterator[None]:
        """Hook ``transformer.img_in`` for the context; the hook comes off however the body ends."""
        handle = transformer.img_in.register_forward_hook(self._project)
        try:
            yield
        finally:
            handle.remove()
            self._armed = False

    @contextmanager
    def armed(self) -> Iterator[None]:
        self._armed = True
        try:
            yield
        finally:
            self._armed = False

    def _project(self, _module: torch.nn.Module, args: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor | None:
        if not self._armed:
            return None
        tokens = args[0]
        target_len, control_width = self._control_tokens.shape[1:]
        if tokens.shape[-1] + control_width != self._weight.shape[1]:
            raise ValueError(
                f"The Krea-2 Control LoRA's input projection takes {self._weight.shape[1]} features, but the latent "
                f"tokens have {tokens.shape[-1]} and the control tokens {control_width}."
            )
        if tokens.shape[1] < target_len:
            raise ValueError(
                f"The Krea-2 Control LoRA has {target_len} control tokens, but the transformer input has only "
                f"{tokens.shape[1]} image tokens."
            )
        control = self._control_tokens.to(device=tokens.device, dtype=tokens.dtype).expand(tokens.shape[0], -1, -1)
        projected = F.linear(
            torch.cat([tokens[:, :target_len], control], dim=-1),
            self._weight.to(device=tokens.device, dtype=tokens.dtype),
            self._bias.to(device=tokens.device, dtype=tokens.dtype),
        )
        if tokens.shape[1] == target_len:
            return projected
        # Joint edit references follow the target through `img_in` and keep the stock projection.
        return torch.cat([projected, output[:, target_len:]], dim=1)
