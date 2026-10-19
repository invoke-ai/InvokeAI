"""Tiling on the FLUX.2 image-to-latents and latents-to-image nodes, and the OOM fallbacks.

The VAE is a real, tiny `AutoencoderKLFlux2` on the CPU: four blocks, so the same 8x compression
`scoped_vae_tiling` derives its geometry from, and the genuine `_tiled_encode`/`tiled_decode`. The
assertions are about tiles that were actually encoded and assembled, not about recorded flags.

**The images are 528x784 on purpose.** Diffusers tiles only when a side is strictly greater than the
tile, so the default 512px tile has to be smaller than the image for a "tiled" cell to take the
tiled path; neither side is a multiple of any tile used here, so the assembly has partial edge tiles.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2

from invokeai.app.invocations.vae.flux2_vae_decode import Flux2VaeDecodeInvocation
from invokeai.app.invocations.vae.flux2_vae_encode import Flux2VaeEncodeInvocation
from invokeai.backend.util.vae_tiling_scope import DEFAULT_TILE_SAMPLE_MIN_SIZE, MIN_TILE_SAMPLE_SIZE
from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_flux2
from invokeai.backend.working_memory.estimate import WorkingMemoryEstimate

ENCODE_MODULE = "invokeai.app.invocations.vae.flux2_vae_encode"
DECODE_MODULE = "invokeai.app.invocations.vae.flux2_vae_decode"
CPU = torch.device("cpu")
IMAGE_SHAPE = (1, 3, 528, 784)
LATENT_SHAPE = (1, 32, 66, 98)


def _tiling_state(vae: AutoencoderKLFlux2) -> tuple:
    return (vae.use_tiling, vae.tile_sample_min_size, vae.tile_latent_min_size, vae.tile_overlap_factor)


@pytest.fixture
def vae() -> AutoencoderKLFlux2:
    """~15k parameters. `sample_size=64` makes the VAE's own idle tile geometry differ from every tile the
    nodes ask for, so a leaked tile shows up as a changed attribute."""
    with torch.random.fork_rng():
        torch.manual_seed(0)
        vae = AutoencoderKLFlux2(
            block_out_channels=(4, 4, 4, 4),
            layers_per_block=1,
            latent_channels=32,
            norm_num_groups=2,
            sample_size=64,
        ).eval()
    # Spies on the tiled paths, and a record of the tiling state each top-level call ran under.
    vae._tiled_encode = MagicMock(wraps=vae._tiled_encode)
    vae.tiled_decode = MagicMock(wraps=vae.tiled_decode)
    vae.seen_tiling = []
    for name in ("encode", "decode"):
        real = getattr(vae, name)

        def record(*args, _real=real, **kwargs):
            vae.seen_tiling.append(_tiling_state(vae))
            return _real(*args, **kwargs)

        setattr(vae, name, MagicMock(side_effect=record))
    return vae


def _fail_first_call_with_oom(vae: AutoencoderKLFlux2, name: str) -> None:
    passthrough = getattr(vae, name).side_effect
    calls = []

    def fail_then_pass(*args, **kwargs):
        calls.append(None)
        result = passthrough(*args, **kwargs)
        if len(calls) == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return result

    getattr(vae, name).side_effect = fail_then_pass


def _context(
    vae: AutoencoderKLFlux2, *, force_tiled_decode: bool = False, auto_tiled_decode: bool = True, latents=None
):
    vae_info = MagicMock()
    vae_info.model = vae
    vae_info.compute_device = CPU
    vae_info.model_on_device.return_value.__enter__.return_value = (None, vae)

    context = MagicMock()
    context.models.load.return_value = vae_info
    # Explicit: a bare MagicMock attribute reads as truthy and would tile everything.
    context.config.get.return_value.force_tiled_decode = force_tiled_decode
    context.config.get.return_value.auto_tiled_decode = auto_tiled_decode
    context.tensors.save.return_value = "latents"
    context.tensors.load.return_value = latents
    context.images.save.side_effect = lambda image: MagicMock(
        image_name="out.png", width=image.width, height=image.height
    )
    return context


def _encode(context, image: torch.Tensor, **fields) -> torch.Tensor:
    invocation = Flux2VaeEncodeInvocation.model_construct(
        image=MagicMock(image_name="in.png"), vae=MagicMock(vae=MagicMock()), **fields
    )
    with patch(f"{ENCODE_MODULE}.image_resized_to_grid_as_tensor", return_value=image[0]):
        invocation.invoke(context)
    return context.tensors.save.call_args.kwargs["tensor"]


def _decode(context, **fields) -> np.ndarray:
    invocation = Flux2VaeDecodeInvocation.model_construct(
        latents=MagicMock(latents_name="latents"), vae=MagicMock(vae=MagicMock()), **fields
    )
    invocation.invoke(context)
    return np.asarray(context.images.save.call_args.kwargs["image"])


@pytest.fixture
def image() -> torch.Tensor:
    return torch.rand(IMAGE_SHAPE, generator=torch.Generator().manual_seed(1)) * 2 - 1


@pytest.fixture
def latents() -> torch.Tensor:
    return torch.randn(LATENT_SHAPE, generator=torch.Generator().manual_seed(2))


def _reservations(context) -> list[WorkingMemoryEstimate]:
    vae_info = context.models.load.return_value
    return [c.kwargs["working_memory"] for c in vae_info.model_on_device.call_args_list]


def _expected_reservation(operation: str, tensor: torch.Tensor, vae, tile_size: int | None) -> WorkingMemoryEstimate:
    return estimate_vae_working_memory_flux2(
        operation=operation, image_tensor=tensor, vae=vae, tile_size=tile_size, device=CPU
    )


class TestEncodeTiling:
    def test_a_tiled_encode_assembles_the_untiled_latent_shape(self, vae, image):
        untiled = _encode(_context(vae), image)
        assert vae._tiled_encode.call_count == 0

        tiled = _encode(_context(vae), image, tiled=True, tile_size=128)

        assert vae._tiled_encode.call_count == 1
        assert tiled.shape == untiled.shape == LATENT_SHAPE
        # The encoder's normalisation and attention are global, so a tiled encode cannot match.
        assert not torch.allclose(tiled, untiled)

    def test_force_tiled_decode_tiles_the_encode_without_the_field(self, vae, image):
        _encode(_context(vae, force_tiled_decode=True), image)

        assert vae._tiled_encode.call_count == 1
        assert vae.seen_tiling[0][1] == DEFAULT_TILE_SAMPLE_MIN_SIZE

    def test_the_shared_vae_keeps_its_tiling_state_after_a_tiled_encode(self, vae, image):
        before = _tiling_state(vae)
        _encode(_context(vae), image, tiled=True, tile_size=256)
        assert _tiling_state(vae) == before

    def test_an_oom_while_already_tiled_is_raised_once_and_the_tiling_state_restored(self, vae, image):
        before = _tiling_state(vae)
        vae._tiled_encode.side_effect = torch.cuda.OutOfMemoryError("CUDA out of memory")
        context = _context(vae)

        with pytest.raises(torch.cuda.OutOfMemoryError):
            _encode(context, image, tiled=True)

        assert vae.encode.call_count == 1
        assert _tiling_state(vae) == before

    def test_an_untiled_oom_retries_once_tiled_and_returns_the_tiled_latent(self, vae, image):
        directly_tiled = _encode(_context(vae), image, tiled=True)
        vae.seen_tiling.clear()
        _fail_first_call_with_oom(vae, "encode")
        context = _context(vae)

        after_retry = _encode(context, image, tiled=False, tile_size=256)

        # The retry takes the default tile, not the field's leftover 256.
        assert [state[0] for state in vae.seen_tiling] == [False, True]
        assert vae.seen_tiling[1][1] == DEFAULT_TILE_SAMPLE_MIN_SIZE
        assert torch.equal(after_retry, directly_tiled)
        assert _reservations(context) == [
            _expected_reservation("encode", image, vae, None),
            _expected_reservation("encode", image, vae, DEFAULT_TILE_SAMPLE_MIN_SIZE),
        ]
        context.util.signal_progress.assert_any_call("VAE encode ran out of memory, retrying tiled")
        assert "not identical to an untiled encode" in context.logger.warning.call_args[0][0]

    @pytest.mark.parametrize(
        "fields,expected_tile",
        [
            ({}, None),
            ({"tiled": True}, DEFAULT_TILE_SAMPLE_MIN_SIZE),
            ({"tiled": True, "tile_size": 256}, 256),
            ({"tiled": True, "tile_size": 64}, MIN_TILE_SAMPLE_SIZE),
        ],
        ids=["untiled", "default-tile", "explicit-tile", "below-minimum-tile"],
    )
    def test_the_reservation_prices_the_encode_that_runs(self, vae, image, fields, expected_tile):
        context = _context(vae)
        _encode(context, image, **fields)
        assert _reservations(context) == [_expected_reservation("encode", image, vae, expected_tile)]


class TestDecodeTiling:
    def test_a_tiled_decode_assembles_the_untiled_image_size(self, vae, latents):
        untiled = _decode(_context(vae, latents=latents))
        assert vae.tiled_decode.call_count == 0

        tiled = _decode(_context(vae, latents=latents), tiled=True, tile_size=128)

        assert vae.tiled_decode.call_count == 1
        assert tiled.shape == untiled.shape == (IMAGE_SHAPE[2], IMAGE_SHAPE[3], 3)
        assert not np.array_equal(tiled, untiled)

    def test_force_tiled_decode_tiles_without_the_field(self, vae, latents):
        _decode(_context(vae, force_tiled_decode=True, latents=latents))

        assert vae.tiled_decode.call_count == 1
        assert vae.seen_tiling[0][1] == DEFAULT_TILE_SAMPLE_MIN_SIZE

    def test_the_shared_vae_keeps_its_tiling_state_after_a_tiled_decode(self, vae, latents):
        before = _tiling_state(vae)
        _decode(_context(vae, latents=latents), tiled=True, tile_size=256)
        assert _tiling_state(vae) == before

    def test_an_oom_while_already_tiled_is_raised_once_and_the_tiling_state_restored(self, vae, latents):
        before = _tiling_state(vae)
        vae.tiled_decode.side_effect = torch.cuda.OutOfMemoryError("CUDA out of memory")

        with pytest.raises(torch.cuda.OutOfMemoryError):
            _decode(_context(vae, latents=latents), tiled=True)

        assert vae.decode.call_count == 1
        assert _tiling_state(vae) == before

    def test_an_untiled_oom_retries_once_tiled_and_returns_the_tiled_image(self, vae, latents):
        directly_tiled = _decode(_context(vae, latents=latents), tiled=True)
        vae.seen_tiling.clear()
        _fail_first_call_with_oom(vae, "decode")
        context = _context(vae, latents=latents)

        after_retry = _decode(context, tiled=False, tile_size=256)

        # The retry takes the default tile, not the field's leftover 256.
        assert [state[0] for state in vae.seen_tiling] == [False, True]
        assert vae.seen_tiling[1][1] == DEFAULT_TILE_SAMPLE_MIN_SIZE
        assert np.array_equal(after_retry, directly_tiled)
        context.util.signal_progress.assert_any_call("VAE decode ran out of memory, retrying tiled")
        assert "not identical to an untiled decode" in context.logger.warning.call_args[0][0]

    @pytest.mark.parametrize(
        "fields,force,expected_tile",
        [
            ({}, False, None),
            ({"tiled": True, "tile_size": 256}, False, 256),
            ({"tile_size": 256}, True, 256),
            ({}, True, DEFAULT_TILE_SAMPLE_MIN_SIZE),
            ({"tiled": True, "tile_size": 64}, False, MIN_TILE_SAMPLE_SIZE),
        ],
        ids=["untiled", "field", "forced-with-tile-size", "forced-default-tile", "below-minimum-tile"],
    )
    def test_the_reservation_prices_the_decode_that_runs(self, vae, latents, fields, force, expected_tile):
        context = _context(vae, force_tiled_decode=force, latents=latents)
        _decode(context, **fields)
        assert _reservations(context) == [_expected_reservation("decode", latents, vae, expected_tile)]

    @pytest.mark.parametrize("auto", [True, False], ids=["auto-tiled-decode-on", "auto-tiled-decode-off"])
    def test_a_decode_too_large_for_its_gpu_is_tiled_up_front_unless_switched_off(self, vae, latents, auto):
        """On Windows an allocation that does not fit is paged rather than refused, so the OOM retry never fires
        and the decision has to be made from the estimate alone."""
        context = _context(vae, auto_tiled_decode=auto, latents=latents)
        with patch(f"{DECODE_MODULE}.should_pretile_vae_decode", return_value=True) as pretile:
            _decode(context)

        untiled_estimate = _expected_reservation("decode", latents, vae, None)
        if auto:
            pretile.assert_called_once_with(CPU, untiled_estimate.reservation_bytes)
            assert vae.tiled_decode.call_count == 1
            assert _reservations(context) == [
                _expected_reservation("decode", latents, vae, DEFAULT_TILE_SAMPLE_MIN_SIZE)
            ]
        else:
            pretile.assert_not_called()
            assert vae.tiled_decode.call_count == 0
            assert _reservations(context) == [untiled_estimate]
