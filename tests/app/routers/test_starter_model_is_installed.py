"""A Qwen3 encoder starter counts as installed when any encoder of its size is.

The Z-Image and FLUX.2 Klein starters depend on a Qwen3 encoder, and every such encoder of one size serves them
alike. Recognising the size only by the source spellings BFL and Tongyi use (`qwen3_4b`, `Z-Image`) left out
Comfy-Org's `qwen_3_4b_fp4_mixed`: installing "Z-Image Turbo (NVFP4)" beside an installed 4B encoder would queue a
redundant 3.5GB download. Comfy's Klein repos spell the size in lower case (`flux-klein-9b`), so their encoders are
recognised by the file name alone.
"""

import pytest

from invokeai.app.api.routers.model_manager import get_is_installed
from invokeai.backend.model_manager.configs.qwen3_encoder import Qwen3Encoder_Checkpoint_Config
from invokeai.backend.model_manager.starter_models.common import (
    flux2_klein_qwen3_4b_encoder_fp4,
    flux2_klein_qwen3_8b_encoder_fp4,
    z_image_qwen3_encoder_fp4,
)
from invokeai.backend.model_manager.starter_models.types import StarterModel
from invokeai.backend.model_manager.taxonomy import Qwen3VariantType

FP4_ENCODER_STARTERS = [
    (z_image_qwen3_encoder_fp4, Qwen3VariantType.Qwen3_4B, Qwen3VariantType.Qwen3_8B),
    (flux2_klein_qwen3_4b_encoder_fp4, Qwen3VariantType.Qwen3_4B, Qwen3VariantType.Qwen3_8B),
    (flux2_klein_qwen3_8b_encoder_fp4, Qwen3VariantType.Qwen3_8B, Qwen3VariantType.Qwen3_4B),
]


def _installed_encoder(variant: Qwen3VariantType) -> Qwen3Encoder_Checkpoint_Config:
    return Qwen3Encoder_Checkpoint_Config.model_construct(
        key="installed",
        name=f"my {variant.value} encoder",
        source=f"D:/models/{variant.value}.safetensors",
        variant=variant,
    )


@pytest.mark.parametrize(
    ("starter", "size", "other_size"), FP4_ENCODER_STARTERS, ids=["z_image_4b", "klein_4b", "klein_8b"]
)
def test_an_fp4_encoder_starter_is_satisfied_by_an_installed_encoder_of_its_size_only(
    starter: StarterModel, size: Qwen3VariantType, other_size: Qwen3VariantType
) -> None:
    assert get_is_installed(starter, [_installed_encoder(size)])
    assert not get_is_installed(starter, [_installed_encoder(other_size)])
