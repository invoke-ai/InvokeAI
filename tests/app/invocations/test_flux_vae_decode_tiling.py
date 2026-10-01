"""Tiling on the FLUX.1 latents-to-image node: the two ways to ask for it, and the OOM fallback.

The node gained `tiled`/`tile_size` when the FLUX.1 VAE became a diffusers `AutoencoderKL`; before
that the app config switch was the only way to reach a tiled decode here, because the node had no
fields and adding them to the hand-rolled class would not have bought the encode side anything.
Both routes are covered, because they are wired separately and either can rot on its own.

The VAE is a real, tiny `AutoencoderKL` (see `conftest.flux_shaped_vae`) rather than a mock, so
`scoped_vae_tiling` runs its genuine state machine and the assertions are about tiling state that
actually existed rather than about calls that were recorded.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from invokeai.app.invocations.vae.flux_vae_decode import FluxVaeDecodeInvocation
from invokeai.backend.util.vae_tiling_scope import DEFAULT_TILE_SAMPLE_MIN_SIZE


def _build_decode_mocks(vae, latents: torch.Tensor, decoded: torch.Tensor, force_tiled_decode: bool = False):
    """Wire FluxVaeDecodeInvocation.invoke to run end-to-end on CPU against a real tiny VAE.

    `decode` is replaced so the test controls the result and can raise, and so the tiling state at
    the moment of the call can be recorded -- that is the thing under test, and it is restored by
    the time `invoke` returns.
    """
    tiling_state: list[dict] = []

    def record_and_return(latents_arg, return_dict=True):
        tiling_state.append(
            {
                "use_tiling": vae.use_tiling,
                "tile_sample_min_size": vae.tile_sample_min_size,
                "tile_latent_min_size": vae.tile_latent_min_size,
            }
        )
        return (decoded,) if return_dict is False else MagicMock(sample=decoded)

    vae.decode = MagicMock(side_effect=record_and_return)

    vae_info = MagicMock()
    vae_info.model = vae
    vae_info.compute_device = torch.device("cpu")
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=(None, vae))
    cm.__exit__ = MagicMock(return_value=None)
    vae_info.model_on_device.return_value = cm

    context = MagicMock()
    context.models.load.return_value = vae_info
    context.tensors.load.return_value = latents
    # A bare MagicMock config would read as truthy and silently tile everything.
    context.config.get.return_value.force_tiled_decode = force_tiled_decode
    image_dto = MagicMock()
    image_dto.image_name = "test.png"
    image_dto.width = decoded.shape[-1]
    image_dto.height = decoded.shape[-2]
    context.images.save.return_value = image_dto
    return vae_info, context, tiling_state


def _build_invocation(tiled: bool = False, tile_size: int = 0) -> FluxVaeDecodeInvocation:
    return FluxVaeDecodeInvocation.model_construct(
        latents=MagicMock(latents_name="test_latents"),
        vae=MagicMock(vae=MagicMock()),
        tiled=tiled,
        tile_size=tile_size,
    )


class TestAskingForTiling:
    def test_the_default_decodes_untiled(self, flux_shaped_vae):
        vae = flux_shaped_vae()
        _, context, state = _build_decode_mocks(vae, torch.zeros(1, 16, 64, 64), torch.zeros(1, 3, 512, 512))
        _build_invocation().invoke(context)
        assert state == [{"use_tiling": False, "tile_sample_min_size": 64, "tile_latent_min_size": 8}]

    def test_the_node_field_reaches_the_tiled_path(self, flux_shaped_vae):
        vae = flux_shaped_vae()
        _, context, state = _build_decode_mocks(vae, torch.zeros(1, 16, 64, 64), torch.zeros(1, 3, 512, 512))
        _build_invocation(tiled=True, tile_size=256).invoke(context)
        assert state[0]["use_tiling"] is True
        assert state[0]["tile_sample_min_size"] == 256

    def test_the_config_flag_reaches_the_tiled_path(self, flux_shaped_vae):
        vae = flux_shaped_vae()
        _, context, state = _build_decode_mocks(
            vae, torch.zeros(1, 16, 64, 64), torch.zeros(1, 3, 512, 512), force_tiled_decode=True
        )
        _build_invocation().invoke(context)
        assert state[0]["use_tiling"] is True
        # 0 is the "model default" sentinel, which resolves to the shipped tile.
        assert state[0]["tile_sample_min_size"] == DEFAULT_TILE_SAMPLE_MIN_SIZE

    def test_the_tiling_state_is_restored_afterwards(self, flux_shaped_vae):
        """The VAE belongs to the model cache; a tiled decode must not leave the next one tiled."""
        vae = flux_shaped_vae()
        _, context, _ = _build_decode_mocks(vae, torch.zeros(1, 16, 64, 64), torch.zeros(1, 3, 512, 512))
        before = (vae.use_tiling, vae.tile_sample_min_size, vae.tile_latent_min_size)
        _build_invocation(tiled=True, tile_size=256).invoke(context)
        assert (vae.use_tiling, vae.tile_sample_min_size, vae.tile_latent_min_size) == before

    @pytest.mark.parametrize(
        "tiled,force_tiled_decode,expected_tile_size",
        [(False, False, None), (True, False, 256), (False, True, 0)],
    )
    def test_the_reservation_matches_the_decode_that_will_run(
        self, flux_shaped_vae, tiled, force_tiled_decode, expected_tile_size
    ):
        """A tiled decode reserved for a single pass is the OOM the retry exists to avoid, and an
        untiled decode reserved for a tile evicts models it did not need to."""
        path = "invokeai.app.invocations.vae.flux_vae_decode.estimate_vae_working_memory_flux"
        vae = flux_shaped_vae()
        _, context, _ = _build_decode_mocks(
            vae, torch.zeros(1, 16, 64, 64), torch.zeros(1, 3, 512, 512), force_tiled_decode=force_tiled_decode
        )
        with patch(path, return_value=1024) as estimate:
            _build_invocation(tiled=tiled, tile_size=256 if tiled else 0).invoke(context)
        assert estimate.call_args.kwargs["tile_size"] == expected_tile_size


class TestOomFallback:
    def test_an_untiled_oom_retries_once_tiled_and_says_so(self, flux_shaped_vae):
        vae = flux_shaped_vae()
        decoded = torch.zeros(1, 3, 512, 512)
        _, context, state = _build_decode_mocks(vae, torch.zeros(1, 16, 64, 64), decoded)
        calls = {"n": 0}
        real = vae.decode.side_effect

        def fail_then_succeed(latents_arg, return_dict=True):
            calls["n"] += 1
            result = real(latents_arg, return_dict=return_dict)
            if calls["n"] == 1:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")
            return result

        vae.decode = MagicMock(side_effect=fail_then_succeed)

        result = _build_invocation().invoke(context)

        assert vae.decode.call_count == 2
        assert state[0]["use_tiling"] is False
        assert state[1]["use_tiling"] is True
        # The retry takes noticeably longer than the failed attempt; the user is told why.
        context.util.signal_progress.assert_any_call("VAE decode ran out of memory, retrying tiled")
        assert result.width == 512

    def test_an_oom_while_already_tiled_reraises(self, flux_shaped_vae):
        vae = flux_shaped_vae()
        _, context, _ = _build_decode_mocks(
            vae, torch.zeros(1, 16, 64, 64), torch.zeros(1, 3, 512, 512), force_tiled_decode=True
        )
        vae.decode = MagicMock(side_effect=torch.cuda.OutOfMemoryError("CUDA out of memory"))

        with pytest.raises(torch.cuda.OutOfMemoryError):
            _build_invocation().invoke(context)

        assert vae.decode.call_count == 1
