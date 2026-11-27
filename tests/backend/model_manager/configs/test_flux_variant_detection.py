"""Tests for FLUX variant detection boundaries."""

import pytest
import torch

from invokeai.backend.model_manager.configs.main import _get_flux_variant
from invokeai.backend.model_manager.taxonomy import FluxVariantType


@pytest.mark.parametrize(
    ("input_key", "channels", "guidance", "expected"),
    [
        ("img_in.weight", 64, False, FluxVariantType.Schnell),
        ("img_in.weight", 64, True, FluxVariantType.Dev),
        ("model.diffusion_model.img_in.weight", 64, True, FluxVariantType.Dev),
        ("img_in.weight", 384, True, FluxVariantType.DevFill),
        ("model.diffusion_model.img_in.weight", 384, True, FluxVariantType.DevFill),
    ],
)
def test_flux_variant_detection_supports_known_key_layouts(
    input_key: str, channels: int, guidance: bool, expected: FluxVariantType
) -> None:
    state_dict = {input_key: torch.empty(1, channels)}
    if guidance:
        guidance_key = (
            "model.diffusion_model.guidance_in.out_layer.weight"
            if input_key.startswith("model.")
            else "guidance_in.out_layer.weight"
        )
        state_dict[guidance_key] = torch.empty(1)

    assert _get_flux_variant(state_dict) is expected


def test_flux_variant_detection_reports_missing_input_channels() -> None:
    assert _get_flux_variant({"double_blocks.0.img_attn.norm.key_norm.scale": torch.empty(1)}) is None
