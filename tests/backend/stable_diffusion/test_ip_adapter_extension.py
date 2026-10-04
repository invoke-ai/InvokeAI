from types import SimpleNamespace

import pytest
import torch

from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningMode
from invokeai.backend.stable_diffusion.extension_callback_type import ExtensionCallbackType
from invokeai.backend.stable_diffusion.extensions.ip_adapter import (
    IPAdapterAttentionWeights,
    IPAdapterExt,
    RegionalIPDataNew,
)
from invokeai.backend.stable_diffusion.extensions_manager import ExtensionsManager


def _make_ip_adapter_extension() -> IPAdapterExt:
    return IPAdapterExt(
        node_context=None,
        model_id=None,
        image_encoder_id=None,
        images=[],
        weight=1.0,
        begin_step_percent=0.0,
        end_step_percent=1.0,
        target_blocks=[],
        method="full",
        mask=None,
    )


def test_multiple_ip_adapter_extensions_build_shared_masks_once():
    ip_data = RegionalIPDataNew(ConditioningMode.Both, torch.device("cpu"), torch.float32)
    ip_data.add(torch.zeros((1, 1, 2)), torch.zeros((1, 1, 2)), torch.zeros((1, 1, 8, 8)))
    ip_data.add(torch.zeros((1, 1, 2)), torch.zeros((1, 1, 2)), torch.ones((1, 1, 8, 8)))
    ctx = SimpleNamespace(extra={"IPAdapterExt_IPData": ip_data})

    ext_manager = ExtensionsManager()
    ext_manager.add_extension(_make_ip_adapter_extension())
    ext_manager.add_extension(_make_ip_adapter_extension())
    ext_manager.run_callback(ExtensionCallbackType.PRE_DENOISE_LOOP, ctx)

    masks = ip_data.get_masks(query_seq_len=64)
    assert masks.shape == (1, 2, 64, 1)
    assert torch.equal(masks[0, 0], torch.zeros((64, 1)))
    assert torch.equal(masks[0, 1], torch.ones((64, 1)))


@pytest.mark.parametrize(
    ("conditioning_mode", "expected"),
    [
        (ConditioningMode.Negative, torch.tensor([[[0.5, 1.5]]])),
        (ConditioningMode.Positive, torch.zeros((1, 1, 2))),
    ],
)
def test_style_precise_sequential_guidance_preserves_all_images(conditioning_mode, expected):
    # Two reference images have distinct projected embeddings. The negative branch should attend to both; the
    # positive branch should receive zero image embeddings for style_precise.
    ip_data = RegionalIPDataNew(ConditioningMode.Both, torch.device("cpu"), torch.float32, max_downscale_factor=1)
    uncond = torch.zeros((2, 1, 2))
    cond = torch.tensor([[[1.0, 0.0]], [[0.0, 3.0]]])
    ip_data.add(uncond, cond, torch.ones((1, 1, 1, 1)))
    ip_data.build_masks()
    ip_data.cond_mode = conditioning_mode

    attention_processor = SimpleNamespace(
        _ip_adapter_attention_weights=[
            IPAdapterAttentionWeights(
                ip_adapter_weights=SimpleNamespace(to_k_ip=torch.nn.Identity(), to_v_ip=torch.nn.Identity()),
                skip=False,
                negative=True,
            )
        ]
    )
    query = torch.zeros((1, 1, 1, 2))
    hidden_states = torch.zeros((1, 1, 2))
    encoder_hidden_states = torch.zeros((1, 1, 2))

    output = IPAdapterExt.run_adapters(
        attention_processor,
        SimpleNamespace(heads=1),
        query,
        hidden_states,
        encoder_hidden_states,
        ip_data,
    )

    torch.testing.assert_close(output, expected)
