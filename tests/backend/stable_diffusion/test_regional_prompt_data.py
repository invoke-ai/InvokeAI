import pytest
import torch

from invokeai.backend.stable_diffusion.diffusion.conditioning_data import Range, TextConditioningRegions
from invokeai.backend.stable_diffusion.diffusion.regional_prompt_data import RegionalPromptData


def test_regional_prompt_data_supports_hidiffusion_raunet_downscale() -> None:
    regions = TextConditioningRegions(
        masks=torch.ones((1, 1, 192, 192), dtype=torch.bool),
        ranges=[Range(start=0, end=4)],
    )
    regional_prompt_data = RegionalPromptData(
        regions=[regions],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    attention_mask = regional_prompt_data.get_cross_attn_mask(query_seq_len=6 * 6, key_seq_len=4)

    assert attention_mask.shape == (1, 6 * 6, 4)
    assert torch.count_nonzero(attention_mask) == 0


def test_regional_prompt_data_rejects_unprepared_hidiffusion_scale() -> None:
    regions = TextConditioningRegions(
        masks=torch.ones((1, 1, 192, 192), dtype=torch.bool),
        ranges=[Range(start=0, end=4)],
    )
    regional_prompt_data = RegionalPromptData(
        regions=[regions],
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_downscale_factor=16,
    )

    with pytest.raises(KeyError):
        regional_prompt_data.get_cross_attn_mask(query_seq_len=6 * 6, key_seq_len=4)
