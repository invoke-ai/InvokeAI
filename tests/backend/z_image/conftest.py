import pytest
import torch
from diffusers import ZImageTransformer2DModel


@pytest.fixture
def tiny_z_image_transformer() -> ZImageTransformer2DModel:
    """The real Z-Image transformer at toy width (64, two heads of 32), in float32 on the CPU.

    Every parameter is drawn from N(0, 0.1): the pad tokens start uninitialized, and the zero-initialized projections
    would hide any attention mask or control hint a test is looking for.
    """
    torch.manual_seed(0)
    model = ZImageTransformer2DModel(
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=16,
        dim=64,
        n_layers=2,
        n_refiner_layers=1,
        n_heads=2,
        n_kv_heads=2,
        norm_eps=1e-05,
        qk_norm=True,
        cap_feat_dim=32,
        rope_theta=256.0,
        t_scale=1000.0,
        axes_dims=[8, 12, 12],
        axes_lens=[128, 32, 32],
    ).eval()
    for param in model.parameters():
        torch.nn.init.normal_(param, std=0.1)
    return model
