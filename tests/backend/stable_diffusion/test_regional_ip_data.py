import pytest
import torch

from invokeai.backend.stable_diffusion.diffusion.regional_ip_data import RegionalIPData


def test_regional_ip_data_supports_hidiffusion_raunet_downscale() -> None:
    regional_ip_data = RegionalIPData(
        image_prompt_embeds=[torch.zeros((1, 1, 4, 8))],
        scales=[1.0],
        masks=[torch.ones((1, 1, 192, 192))],
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    masks = regional_ip_data.get_masks(query_seq_len=6 * 6)

    assert masks.shape == (1, 1, 6 * 6, 1)
    assert torch.count_nonzero(masks) == masks.numel()


def test_regional_ip_data_rejects_unprepared_hidiffusion_scale() -> None:
    regional_ip_data = RegionalIPData(
        image_prompt_embeds=[torch.zeros((1, 1, 4, 8))],
        scales=[1.0],
        masks=[torch.ones((1, 1, 192, 192))],
        dtype=torch.float32,
        device=torch.device("cpu"),
        max_downscale_factor=16,
    )

    with pytest.raises(KeyError):
        regional_ip_data.get_masks(query_seq_len=6 * 6)
