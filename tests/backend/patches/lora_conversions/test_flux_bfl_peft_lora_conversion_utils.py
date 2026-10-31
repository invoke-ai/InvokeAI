from typing import Callable

import pytest
import torch
from diffusers.models.normalization import AdaLayerNormContinuous

from invokeai.backend.flux.modules.layers import LastLayer
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.lora_conversions.flux_bfl_peft_lora_conversion_utils import (
    convert_bfl_lora_patch_to_diffusers,
    lora_model_from_flux2_bfl_peft_state_dict,
    lora_model_from_flux_bfl_peft_state_dict,
)
from invokeai.backend.patches.lora_conversions.flux_lora_constants import FLUX_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw

HIDDEN_SIZE = 8
OUT_CHANNELS = 4
RANK = 2


class _Flux2FinalLayer(torch.nn.Module):
    """The tail of Flux2Transformer2DModel: AdaLayerNormContinuous, then the output projection."""

    def __init__(self):
        super().__init__()
        self.norm_out = AdaLayerNormContinuous(HIDDEN_SIZE, HIDDEN_SIZE, elementwise_affine=False, eps=1e-6)
        self.proj_out = torch.nn.Linear(HIDDEN_SIZE, OUT_CHANNELS)

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        return self.proj_out(self.norm_out(x, vec))


def _flux2_state_dict_converter(state_dict: dict[str, torch.Tensor]) -> ModelPatchRaw:
    return lora_model_from_flux2_bfl_peft_state_dict(state_dict)


def _flux1_converter_then_apply_time_conversion(state_dict: dict[str, torch.Tensor]) -> ModelPatchRaw:
    # What Flux2DenoiseInvocation does with a BFL LoRA that was loaded by the FLUX.1 converter.
    return convert_bfl_lora_patch_to_diffusers(lora_model_from_flux_bfl_peft_state_dict(state_dict))


@torch.no_grad()
@pytest.mark.parametrize(
    "convert",
    [_flux2_state_dict_converter, _flux1_converter_then_apply_time_conversion],
    ids=["state-dict", "built-patch"],
)
def test_final_layer_modulation_lora_patches_flux2_norm_out(
    convert: Callable[[dict[str, torch.Tensor]], ModelPatchRaw],
) -> None:
    """A LoRA on BFL's final_layer.adaLN_modulation.1, which orders its output (shift, scale), must patch the FLUX.2
    model's norm_out.linear, which orders it (scale, shift), so that the model computes what the LoRA does on the BFL
    model (#10022).
    """
    torch.manual_seed(0)
    bfl_final_layer = LastLayer(HIDDEN_SIZE, patch_size=1, out_channels=OUT_CHANNELS)
    flux2_model = _Flux2FinalLayer()
    # The base weights as the FLUX.2 loader converts them: BFL's halves swapped.
    flux2_model.proj_out.load_state_dict(bfl_final_layer.linear.state_dict())
    bfl_modulation = bfl_final_layer.adaLN_modulation[1]
    for name in ("weight", "bias"):
        bfl_param = getattr(bfl_modulation, name)
        getattr(flux2_model.norm_out.linear, name).copy_(torch.cat(bfl_param.chunk(2, dim=0)[::-1], dim=0))

    x = torch.randn(2, 3, HIDDEN_SIZE)
    vec = torch.randn(2, HIDDEN_SIZE)
    unpatched = flux2_model(x, vec)
    assert torch.allclose(unpatched, bfl_final_layer(x, vec), atol=1e-5)

    down = torch.randn(RANK, HIDDEN_SIZE)
    up = torch.randn(2 * HIDDEN_SIZE, RANK)
    state_dict = {
        "diffusion_model.final_layer.adaLN_modulation.1.lora_A.weight": down,
        "diffusion_model.final_layer.adaLN_modulation.1.lora_B.weight": up,
        # A block layer, which marks the LoRA as BFL-keyed for the apply-time conversion.
        "diffusion_model.double_blocks.0.img_attn.proj.lora_A.weight": torch.randn(RANK, HIDDEN_SIZE),
        "diffusion_model.double_blocks.0.img_attn.proj.lora_B.weight": torch.randn(HIDDEN_SIZE, RANK),
    }
    patch = convert(state_dict)
    final_layer_key = f"{FLUX_LORA_TRANSFORMER_PREFIX}norm_out.linear"
    assert set(patch.layers) == {final_layer_key, f"{FLUX_LORA_TRANSFORMER_PREFIX}transformer_blocks.0.attn.to_out.0"}

    with LayerPatcher.apply_smart_model_patches(
        model=flux2_model,
        patches=[(ModelPatchRaw(layers={final_layer_key: patch.layers[final_layer_key]}), 1.0)],
        prefix=FLUX_LORA_TRANSFORMER_PREFIX,
        dtype=torch.float32,
        force_direct_patching=True,
    ):
        patched = flux2_model(x, vec)

    bfl_modulation.weight += up @ down
    expected = bfl_final_layer(x, vec)
    assert not torch.allclose(expected, unpatched, atol=1e-2)
    assert torch.allclose(patched, expected, atol=1e-5)
