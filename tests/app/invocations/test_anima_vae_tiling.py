"""Tiling state on the Anima VAE nodes' shared, cached Wan VAE.

`enable_tiling` writes the tile geometry onto the module and `disable_tiling` restores only the
flag, never the sizes -- and the module is the model cache's own instance, shared with the
Qwen-Image nodes whenever a native-layout `qwen_image_vae` single file is loaded. A real
`AutoencoderKLWan` is used here rather than a mock precisely because that write-through is the
behaviour under test; a mocked `enable_tiling` would leave nothing behind to leak.
"""

from unittest.mock import MagicMock, patch

import torch
from diffusers.models.autoencoders import AutoencoderKLWan

from invokeai.app.invocations.vae.anima_image_to_latents import AnimaImageToLatentsInvocation
from invokeai.app.invocations.vae.anima_latents_to_image import (
    ANIMA_VAE_TILE_SIZE,
    ANIMA_VAE_TILE_STRIDE,
    AnimaLatentsToImageInvocation,
)
from invokeai.backend.util.devices import TorchDevice


def _build_tiny_vae() -> AutoencoderKLWan:
    """The smallest Wan VAE the node accepts: the Qwen-Image geometry (16 latent channels, 8x spatial,
    unpatchified) at minimal width."""
    return AutoencoderKLWan(
        base_dim=2,
        z_dim=16,
        dim_mult=[1, 1, 1, 1],
        num_res_blocks=1,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        latents_mean=[0.0] * 16,
        latents_std=[1.0] * 16,
    ).eval()


def _tiling_state(vae: AutoencoderKLWan) -> tuple:
    return (
        vae.use_tiling,
        vae.tile_sample_min_height,
        vae.tile_sample_min_width,
        vae.tile_sample_stride_height,
        vae.tile_sample_stride_width,
    )


def _build_context(vae: AutoencoderKLWan, latents: torch.Tensor):
    vae_info = MagicMock()
    vae_info.model = vae
    # The invocation places latents on the VAE's intended compute device (see #9373), so this must
    # be a real torch.device rather than a MagicMock for `latents.to(device=...)` to work.
    vae_info.compute_device = torch.device("cpu")
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=(None, vae))
    cm.__exit__ = MagicMock(return_value=None)
    vae_info.model_on_device.return_value = cm

    context = MagicMock()
    context.models.load.return_value = vae_info
    context.tensors.load.return_value = latents
    image_dto = MagicMock()
    image_dto.image_name = "test.png"
    image_dto.width = latents.shape[-1] * 8
    image_dto.height = latents.shape[-2] * 8
    context.images.save.return_value = image_dto
    return context


def _build_invocation() -> AnimaLatentsToImageInvocation:
    return AnimaLatentsToImageInvocation.model_construct(
        latents=MagicMock(latents_name="test_latents"),
        vae=MagicMock(vae=MagicMock()),
    )


def test_the_oom_retry_does_not_leave_the_shared_vae_tiled():
    """The leak the scoped helper exists for, on the path that used to set the geometry twice.

    Without the scope the retry leaves `use_tiling` set and the 512/384 geometry written onto the
    cached module, so the next node to decode through it silently tiles at a geometry it never
    asked for -- including the Qwen-Image nodes, whose stock tile is 256/192.
    """
    vae = _build_tiny_vae()
    before = _tiling_state(vae)
    context = _build_context(vae, torch.zeros(1, 16, 8, 8))

    real_decode = vae.decode
    attempts: list[bool] = []

    def flaky_decode(*args, **kwargs):
        attempts.append(vae.use_tiling)
        if len(attempts) == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 5.9 GiB")
        return real_decode(*args, **kwargs)

    vae.decode = flaky_decode
    with patch.object(TorchDevice, "choose_torch_device", return_value=torch.device("cpu")):
        _build_invocation().invoke(context)

    # The first attempt was untiled and the retry was tiled -- otherwise nothing was restored.
    assert attempts == [False, True]
    assert vae.use_tiling is False
    assert _tiling_state(vae) == before


def test_a_tiled_decode_applies_the_calibrated_geometry_and_restores_it():
    """The tiling decision is made against a working-memory estimate calibrated at 512/384, so the
    decode has to run at that geometry -- and hand the cached module back unchanged."""
    vae = _build_tiny_vae()
    before = _tiling_state(vae)
    context = _build_context(vae, torch.zeros(1, 16, 8, 8))

    real_decode = vae.decode
    during: list[tuple] = []

    def spy_decode(*args, **kwargs):
        during.append(_tiling_state(vae))
        return real_decode(*args, **kwargs)

    vae.decode = spy_decode
    with (
        patch.object(TorchDevice, "choose_torch_device", return_value=torch.device("cpu")),
        patch.object(AnimaLatentsToImageInvocation, "_use_tiled_decode", return_value=True),
    ):
        _build_invocation().invoke(context)

    assert during == [(True, ANIMA_VAE_TILE_SIZE, ANIMA_VAE_TILE_SIZE, ANIMA_VAE_TILE_STRIDE, ANIMA_VAE_TILE_STRIDE)]
    assert _tiling_state(vae) == before


def _build_encode_context(vae: AutoencoderKLWan):
    vae_info = MagicMock()
    vae_info.model = vae
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=(None, vae))
    cm.__exit__ = MagicMock(return_value=None)
    vae_info.model_on_device.return_value = cm

    context = MagicMock()
    context.models.load.return_value = vae_info
    context.images.get_pil.return_value = MagicMock()
    context.config.get.return_value.force_tiled_decode = False
    context.tensors.save.return_value = "test_latents"
    return vae_info, context


def _encode(context, vae, image: torch.Tensor, tiled: bool = False, tile_size: int = 0):
    """Run the encode node end to end, recording the tiling state the encode actually saw."""
    seen: list[tuple] = []
    real_encode = vae.encode

    def record(x, return_dict=True):
        seen.append(_tiling_state(vae))
        return real_encode(x, return_dict=return_dict)

    vae.encode = record
    invocation = AnimaImageToLatentsInvocation.model_construct(
        image=MagicMock(image_name="test.png"),
        vae=MagicMock(vae=MagicMock()),
        tiled=tiled,
        tile_size=tile_size,
    )
    with (
        patch(
            "invokeai.app.invocations.vae.anima_image_to_latents.image_resized_to_grid_as_tensor",
            return_value=image[0],
        ),
        patch.object(TorchDevice, "choose_torch_device", return_value=torch.device("cpu")),
    ):
        invocation.invoke(context)
    return seen


def test_the_encode_is_untiled_by_default_even_on_a_vae_left_tiled():
    """The node used to call `disable_tiling()` unconditionally, with "encode untiled for exactness"
    as the reason. That reason still holds for ordinary generation, so it is still the default --
    what changed is that it is no longer the only option.

    The VAE arrives with tiling *on*, which is the situation the production comment describes
    ("shared with the decode invocation, which may have enabled tiling"). Starting from a fresh
    instance would assert a value the fixture already handed over.
    """
    vae = _build_tiny_vae()
    vae.enable_tiling(
        tile_sample_min_height=256,
        tile_sample_min_width=256,
        tile_sample_stride_height=192,
        tile_sample_stride_width=192,
    )
    before = _tiling_state(vae)
    _, context = _build_encode_context(vae)

    seen = _encode(context, vae, torch.zeros(1, 3, 768, 768))

    assert seen[0][0] is False
    assert _tiling_state(vae) == before


def test_the_encode_field_applies_the_geometry_and_restores_it():
    vae = _build_tiny_vae()
    _, context = _build_encode_context(vae)
    before = _tiling_state(vae)

    seen = _encode(context, vae, torch.zeros(1, 3, 768, 768), tiled=True, tile_size=ANIMA_VAE_TILE_SIZE)

    assert seen[0] == (
        True,
        ANIMA_VAE_TILE_SIZE,
        ANIMA_VAE_TILE_SIZE,
        ANIMA_VAE_TILE_STRIDE,
        ANIMA_VAE_TILE_STRIDE,
    )
    # The instance is the cache's, shared with the decode node and with the Qwen-Image nodes.
    assert _tiling_state(vae) == before


def test_the_reservation_matches_the_encode_that_will_run():
    vae = _build_tiny_vae()
    _, context = _build_encode_context(vae)
    path = "invokeai.app.invocations.vae.anima_image_to_latents.estimate_vae_working_memory_anima"

    with patch(path, return_value=1024) as estimate:
        _encode(context, vae, torch.zeros(1, 3, 768, 768), tiled=True, tile_size=ANIMA_VAE_TILE_SIZE)
    assert estimate.call_args.kwargs["tile_size"] == ANIMA_VAE_TILE_SIZE

    vae = _build_tiny_vae()
    _, context = _build_encode_context(vae)
    with patch(path, return_value=1024) as estimate:
        _encode(context, vae, torch.zeros(1, 3, 768, 768))
    assert estimate.call_args.kwargs["tile_size"] is None
