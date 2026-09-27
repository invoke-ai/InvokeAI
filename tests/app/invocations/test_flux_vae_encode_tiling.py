"""Tiling on the FLUX.1 image-to-latents node, and the OOM fallback that makes it worth having.

An untiled encode of the frame an upscale graph produces is not merely large, it is quadratic: the
FLUX autoencoder's mid-block attention is a single head over the whole latent grid, so at 3072px it
reserves 18.8 GiB on a 4090 where a tiled encode reserves 0.81 GiB and finishes faster. On a 16 GiB
card the untiled pass does not complete at all.

The node had no tiling of any kind before the FLUX.1 VAE became a diffusers `AutoencoderKL` -- the
hand-rolled class implemented a tiled decode and never a tiled encode, and `encode()` did not so
much as consult the flag.

The VAE is a real, tiny `AutoencoderKL` (`conftest.flux_shaped_vae`), so the tiling state asserted
here is state that actually existed rather than a recorded call.

**The images are 768px on purpose.** `AutoencoderKL._encode` tiles only when a side is strictly
greater than `tile_sample_min_size`, so a 512px image against the default 512px tile takes the
single-pass path -- and a cell named for the tiled path would assert a flag while testing the
untiled one.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from invokeai.app.invocations.vae.flux_vae_encode import FluxVaeEncodeInvocation
from invokeai.backend.util.vae_tiling_scope import DEFAULT_TILE_SAMPLE_MIN_SIZE


@pytest.fixture(autouse=True)
def _pin_to_cpu(monkeypatch):
    """The node moves its input to `choose_torch_device()`, which is CUDA on a machine that has one,
    while the VAE under test stays on the CPU."""
    monkeypatch.setattr(
        "invokeai.backend.util.devices.TorchDevice.choose_torch_device", staticmethod(lambda: torch.device("cpu"))
    )


def _build_encode_mocks(vae, image: torch.Tensor, force_tiled_decode: bool = False):
    tiling_state: list[dict] = []
    real_encode = vae.encode

    def record_and_encode(x, return_dict=True):
        tiling_state.append({"use_tiling": vae.use_tiling, "tile_sample_min_size": vae.tile_sample_min_size})
        return real_encode(x, return_dict=return_dict)

    vae.encode = MagicMock(side_effect=record_and_encode)

    vae_info = MagicMock()
    vae_info.model = vae
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=(None, vae))
    cm.__exit__ = MagicMock(return_value=None)
    vae_info.model_on_device.return_value = cm

    context = MagicMock()
    context.models.load.return_value = vae_info
    context.images.get_pil.return_value = MagicMock()
    # A bare MagicMock config would read as truthy and silently tile everything.
    context.config.get.return_value.force_tiled_decode = force_tiled_decode
    # `LatentsOutput.build` validates this as a string.
    context.tensors.save.return_value = "test_latents"
    return vae_info, context, tiling_state


def _invoke(context, image: torch.Tensor, tiled: bool = False, tile_size: int = 0):
    invocation = FluxVaeEncodeInvocation.model_construct(
        image=MagicMock(image_name="test.png"),
        vae=MagicMock(vae=MagicMock()),
        tiled=tiled,
        tile_size=tile_size,
    )
    with patch("invokeai.app.invocations.vae.flux_vae_encode.image_resized_to_grid_as_tensor", return_value=image[0]):
        return invocation.invoke(context)


class TestAskingForTiling:
    def test_the_default_encodes_untiled(self, flux_shaped_vae):
        vae = flux_shaped_vae()
        _, context, state = _build_encode_mocks(vae, torch.zeros(1, 3, 768, 768))
        _invoke(context, torch.zeros(1, 3, 768, 768))
        assert state[0]["use_tiling"] is False

    def test_the_node_field_reaches_the_tiled_path(self, flux_shaped_vae):
        vae = flux_shaped_vae()
        _, context, state = _build_encode_mocks(vae, torch.zeros(1, 3, 768, 768))
        _invoke(context, torch.zeros(1, 3, 768, 768), tiled=True, tile_size=256)
        assert state[0]["use_tiling"] is True
        assert state[0]["tile_sample_min_size"] == 256

    def test_the_config_flag_reaches_the_tiled_path(self, flux_shaped_vae):
        """`force_tiled_decode` is named for the decode but every encode node already reads it --
        `image_to_latents` and `qwen_image_i2l` both do, and a second setting would say nothing new."""
        vae = flux_shaped_vae()
        _, context, state = _build_encode_mocks(vae, torch.zeros(1, 3, 768, 768), force_tiled_decode=True)
        _invoke(context, torch.zeros(1, 3, 768, 768))
        assert state[0]["use_tiling"] is True
        assert state[0]["tile_sample_min_size"] == DEFAULT_TILE_SAMPLE_MIN_SIZE

    def test_the_tiling_state_is_restored_afterwards(self, flux_shaped_vae):
        """The VAE is the cache's, shared with the decode node, Z-Image, PiD and the ControlNet
        extension -- most of which never touch the flag and would silently inherit it."""
        vae = flux_shaped_vae()
        _, context, _ = _build_encode_mocks(vae, torch.zeros(1, 3, 768, 768))
        before = (vae.use_tiling, vae.tile_sample_min_size, vae.tile_latent_min_size)
        _invoke(context, torch.zeros(1, 3, 768, 768), tiled=True, tile_size=256)
        assert (vae.use_tiling, vae.tile_sample_min_size, vae.tile_latent_min_size) == before

    @pytest.mark.parametrize("tiled,tile_size,expected", [(False, 0, None), (True, 256, 256), (True, 0, 0)])
    def test_the_reservation_matches_the_encode_that_will_run(self, flux_shaped_vae, tiled, tile_size, expected):
        vae = flux_shaped_vae()
        _, context, _ = _build_encode_mocks(vae, torch.zeros(1, 3, 768, 768))
        path = "invokeai.app.invocations.vae.flux_vae_encode.estimate_vae_working_memory_flux"
        with patch(path, return_value=1024) as estimate:
            _invoke(context, torch.zeros(1, 3, 768, 768), tiled=tiled, tile_size=tile_size)
        assert estimate.call_args.kwargs["tile_size"] == expected


class TestTheCallersOutsideThisNode:
    def test_the_shared_helper_defaults_to_untiled(self, flux_shaped_vae):
        """`flux_denoise`, the InstantX ControlNet extension and PiD Upscale all call `vae_encode`
        positionally with no tiling arguments. They pass conditioning images at generation size, so
        the default has to stay what it was."""
        vae = flux_shaped_vae()
        vae_info, _, state = _build_encode_mocks(vae, torch.zeros(1, 3, 256, 256))

        FluxVaeEncodeInvocation.vae_encode(vae_info=vae_info, image_tensor=torch.zeros(1, 3, 256, 256))

        assert state[0]["use_tiling"] is False


class TestOomFallback:
    def test_an_untiled_oom_retries_once_tiled_and_says_so(self, flux_shaped_vae):
        vae = flux_shaped_vae()
        _, context, state = _build_encode_mocks(vae, torch.zeros(1, 3, 768, 768))
        calls = {"n": 0}
        record = vae.encode.side_effect

        def fail_then_succeed(x, return_dict=True):
            calls["n"] += 1
            result = record(x, return_dict=return_dict)
            if calls["n"] == 1:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")
            return result

        vae.encode = MagicMock(side_effect=fail_then_succeed)

        _invoke(context, torch.zeros(1, 3, 768, 768))

        assert vae.encode.call_count == 2
        assert [call["use_tiling"] for call in state] == [False, True]
        context.util.signal_progress.assert_any_call("VAE encode ran out of memory, retrying tiled")
        assert "not identical to an untiled encode" in context.logger.warning.call_args[0][0]

    def test_the_retry_draws_the_same_noise_as_a_directly_tiled_encode(self, flux_shaped_vae):
        """The encode samples from the latent distribution with a fixed seed. If the generator were
        created by the caller rather than inside `vae_encode`, the attempt that failed would already
        have drawn from it, and the retry would return a different latent than asking for tiling up
        front -- a difference nothing downstream could attribute to an OOM that was handled."""
        image = torch.rand(1, 3, 768, 768) * 2 - 1

        vae = flux_shaped_vae()
        _, retry_context, _ = _build_encode_mocks(vae, image)
        record = vae.encode.side_effect
        calls = {"n": 0}

        def fail_then_succeed(x, return_dict=True):
            calls["n"] += 1
            result = record(x, return_dict=return_dict)
            if calls["n"] == 1:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")
            return result

        vae.encode = MagicMock(side_effect=fail_then_succeed)
        _invoke(retry_context, image)

        # The same weights, so any difference in the result is the noise draw and nothing else.
        direct_vae = flux_shaped_vae()
        direct_vae.load_state_dict(vae.state_dict())
        _, direct_context, _ = _build_encode_mocks(direct_vae, image)
        _invoke(direct_context, image, tiled=True, tile_size=0)

        after_retry = retry_context.tensors.save.call_args.kwargs["tensor"]
        directly_tiled = direct_context.tensors.save.call_args.kwargs["tensor"]
        assert torch.equal(after_retry, directly_tiled)

    def test_an_oom_while_already_tiled_reraises(self, flux_shaped_vae):
        vae = flux_shaped_vae()
        _, context, state = _build_encode_mocks(vae, torch.zeros(1, 3, 768, 768), force_tiled_decode=True)
        record = vae.encode.side_effect

        def record_then_fail(x, return_dict=True):
            record(x, return_dict=return_dict)
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")

        vae.encode = MagicMock(side_effect=record_then_fail)

        with pytest.raises(torch.cuda.OutOfMemoryError):
            _invoke(context, torch.zeros(1, 3, 768, 768))

        assert vae.encode.call_count == 1
        assert [call["use_tiling"] for call in state] == [True]
