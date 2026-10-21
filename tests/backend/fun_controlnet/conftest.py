import pytest
import torch
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

from invokeai.backend.fun_controlnet.qwen_image import QwenImageFunControlBranch

TINY_HIDDEN = 16
TINY_HEADS = 2
TINY_HEAD_DIM = 8
TINY_LAYERS = 4
TINY_TEXT_DIM = 24
TINY_CONTROL_LAYERS = (0, 2)


@pytest.fixture
def tiny_qwen_image_transformer() -> QwenImageTransformer2DModel:
    """The real Qwen-Image transformer at toy width (16, two heads of 8), four blocks, in float32 on the CPU.

    Every parameter is drawn from N(0, 0.1): the default init leaves the modulation gates near zero, which would hide
    a hint a test is looking for.
    """
    torch.manual_seed(0)
    model = QwenImageTransformer2DModel(
        patch_size=2,
        in_channels=64,
        out_channels=16,
        num_layers=TINY_LAYERS,
        attention_head_dim=TINY_HEAD_DIM,
        num_attention_heads=TINY_HEADS,
        joint_attention_dim=TINY_TEXT_DIM,
        axes_dims_rope=(2, 4, 2),
    ).eval()
    for param in model.parameters():
        torch.nn.init.normal_(param, std=0.1)
    return model


@pytest.fixture
def tiny_branch() -> QwenImageFunControlBranch:
    """Two control blocks for the tiny transformer, their hints landing after base blocks 0 and 2."""
    torch.manual_seed(1)
    branch = QwenImageFunControlBranch(
        hidden_size=TINY_HIDDEN,
        num_attention_heads=TINY_HEADS,
        attention_head_dim=TINY_HEAD_DIM,
        control_layers=TINY_CONTROL_LAYERS,
    ).eval()
    for param in branch.parameters():
        torch.nn.init.normal_(param, std=0.1)
    return branch
