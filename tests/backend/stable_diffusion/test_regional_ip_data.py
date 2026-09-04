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

    masks = regional_ip_data.get_masks(query_seq_len=12 * 12)

    assert masks.shape == (1, 1, 12 * 12, 1)
    assert torch.count_nonzero(masks) == masks.numel()
