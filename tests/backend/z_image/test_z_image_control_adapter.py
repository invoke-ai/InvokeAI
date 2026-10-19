"""Z-Image control blocks hand back each hint on its own instead of stacking it onto the running state.

Upstream (VideoX-Fun) returns ``torch.stack([hint_0, ..., hint_k, state])`` from every block and unbinds it again in
the next. Each stack is a fresh allocation larger than any the caching allocator has freed, so a 15-block adapter
reserved 136 hint-sized tensors at 1024px -- 4.5 GiB -- where separate hints need 19. The rewrite must not change a
single hint.
"""

import torch
from diffusers.models.transformers.transformer_z_image import ZImageTransformerBlock

from invokeai.backend.z_image.z_image_control_adapter import ZImageControlAdapter
from invokeai.backend.z_image.z_image_controlnet_extension import (
    ZImageControlNetExtension,
    z_image_forward_with_control,
)

LATENT_SIDE = 16


def _upstream_hints(adapter: ZImageControlAdapter, c, x, attn_mask, freqs_cis, adaln_input) -> tuple[torch.Tensor, ...]:
    """The stacked formulation, as VideoX-Fun's `ZImageControlTransformerBlock.forward` and its caller wrote it."""
    for layer in adapter.control_layers:
        if layer.block_id == 0:
            c = layer.before_proj(c) + x
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)
        c = ZImageTransformerBlock.forward(layer, c, attn_mask=attn_mask, freqs_cis=freqs_cis, adaln_input=adaln_input)
        all_c += [layer.after_proj(c), c]
        c = torch.stack(all_c)
    return torch.unbind(c)[:-1]


@torch.no_grad()
def test_the_control_forward_computes_the_stacked_formulations_hints(tiny_z_image_transformer) -> None:
    """Driven through `z_image_forward_with_control`, the path the denoise node takes, with the padding mask it builds
    and a caption that needs padding. The first control block's inputs are captured and replayed through the upstream
    formulation."""
    model = tiny_z_image_transformer
    adapter = ZImageControlAdapter(
        num_control_blocks=3,
        control_in_dim=16,
        dim=model.config.dim,
        n_refiner_layers=1,
        n_heads=model.config.n_heads,
        n_kv_heads=model.config.n_kv_heads,
    ).eval()
    for param in adapter.parameters():
        torch.nn.init.normal_(param, std=0.1)
    extension = ZImageControlNetExtension(
        control_adapter=adapter, control_cond=torch.randn(16, 1, LATENT_SIDE, LATENT_SIDE)
    )

    first_block_inputs = {}
    adapter.control_layers[0].register_forward_pre_hook(
        lambda module, args, kwargs: first_block_inputs.update(c=args[0], **kwargs), with_kwargs=True
    )
    computed = []
    compute_hints = extension.compute_hints

    def recording_compute_hints(**kwargs):
        computed.append(compute_hints(**kwargs))
        return computed[-1]

    extension.compute_hints = recording_compute_hints

    z_image_forward_with_control(
        transformer=model,
        x=[torch.randn(16, 1, LATENT_SIDE, LATENT_SIDE)],
        t=torch.full((1,), 0.5),
        cap_feats=[torch.randn(20, model.config.cap_feat_dim)],
        control_extension=extension,
    )

    assert first_block_inputs["attn_mask"] is not None
    expected = _upstream_hints(adapter, **first_block_inputs)
    (hints,) = computed
    assert len(hints) == len(expected) == 3
    for hint, reference in zip(hints, expected, strict=True):
        assert torch.equal(hint, reference)
