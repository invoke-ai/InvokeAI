"""The Kontext reference encode: its normalisation, and the fallback that makes it survivable.

Two things about this path are easy to get wrong and impossible to notice afterwards.

**The normalisation is hand-written here.** The FLUX.1 VAE used to be InvokeAI's own class, whose
`encode(sample=False)` applied `scale * (raw - shift)` internally. It is a diffusers `AutoencoderKL`
now, which leaves that to the caller, so this extension re-derives it inline. A dropped shift or a
swapped order offsets every reference latent by ~0.042 in latent space: reference adherence degrades
and nothing raises.

**Nothing on this path resizes.** `flux_kontext.py` only wraps the `ImageField`; the reference is
encoded at whatever resolution the user supplied. At 3072px that reserves 19.3 GiB, which no
consumer card has spare beside a resident transformer -- so the encode carries an OOM fallback, and
the fallback has to produce a usable latent rather than merely not crash.
"""

from unittest.mock import MagicMock, Mock

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from PIL import Image

import invokeai.backend.flux.extensions.kontext_extension as kontext_extension
from invokeai.backend.flux.extensions.kontext_extension import KontextExtension


def _tiny_vae() -> AutoencoderKL:
    """FLUX-shaped: four blocks for the real 8x compression, and the FLUX normalisation constants."""
    return AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(4, 4, 4, 4),
        layers_per_block=1,
        latent_channels=16,
        norm_num_groups=2,
        sample_size=64,
        scaling_factor=0.3611,
        shift_factor=0.1159,
        use_quant_conv=False,
        use_post_quant_conv=False,
    ).eval()


def _extension(
    vae: AutoencoderKL,
    image: Image.Image,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> KontextExtension:
    """Build the extension without a model cache, and let `__init__` do the encode."""
    vae_info = MagicMock()
    vae_info.model = vae
    vae_info.compute_device = torch.device("cpu")
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=(None, vae))
    cm.__exit__ = MagicMock(return_value=None)
    vae_info.model_on_device.return_value = cm

    context = MagicMock()
    context.models.load.return_value = vae_info
    context.images.get_pil.return_value = image

    return KontextExtension(
        kontext_conditioning=[MagicMock(image=MagicMock(image_name="reference.png"))],
        context=context,
        vae_field=MagicMock(),
        device=device,
        dtype=dtype,
    )


@pytest.fixture(autouse=True)
def _pin_to_cpu(monkeypatch):
    monkeypatch.setattr(
        "invokeai.backend.util.devices.TorchDevice.choose_torch_device", staticmethod(lambda: torch.device("cpu"))
    )


def _reference_image(edge: int = 64) -> Image.Image:
    torch.manual_seed(0)
    pixels = (torch.rand(edge, edge, 3) * 255).byte().numpy()
    return Image.fromarray(pixels, mode="RGB")


def test_cpu_vae_encodes_kontext_before_moving_conditioning_to_the_transformer(monkeypatch):
    vae = _tiny_vae()
    real_encode = vae.encode

    def encode(pixels):
        assert pixels.device == torch.device("cpu")
        assert pixels.dtype == vae.dtype
        return real_encode(pixels)

    estimator = Mock(wraps=kontext_extension.estimate_vae_working_memory_flux)
    monkeypatch.setattr(kontext_extension, "estimate_vae_working_memory_flux", estimator)
    monkeypatch.setattr(vae, "encode", encode)
    session_device = torch.device("meta")
    monkeypatch.setattr(kontext_extension.TorchDevice, "choose_torch_device", lambda: session_device)

    extension = _extension(vae, _reference_image(), device=session_device, dtype=torch.float16)

    assert estimator.call_args.kwargs["device"] == torch.device("cpu")
    assert estimator.call_args.kwargs["image_tensor"].device == torch.device("cpu")
    assert extension.kontext_latents.device == extension.kontext_ids.device == session_device
    assert extension.kontext_latents.dtype == extension.kontext_ids.dtype == torch.float16


class TestTheNormalisation:
    def test_it_matches_the_convention_the_deleted_class_applied_internally(self):
        """`scale * (raw - shift)` on the distribution mode, which is what `sample=False` returned."""
        vae = _tiny_vae()
        image = _reference_image()

        extension = _extension(vae, image)

        # Recompute it from the VAE directly, the way the deleted class did in one step.
        tensor = torch.from_numpy(torch.tensor(list(image.tobytes()), dtype=torch.uint8).numpy())
        tensor = tensor.reshape(64, 64, 3).permute(2, 0, 1).float() / 255.0
        tensor = (tensor * 2.0 - 1.0).unsqueeze(0)
        with torch.no_grad():
            raw = vae.encode(tensor).latent_dist.mode()
        expected = (raw - vae.config.shift_factor) * vae.config.scaling_factor

        # `kontext_latents` is packed 2x2; compare against the same packing of the expectation.
        from invokeai.backend.flux.sampling_utils import pack

        assert torch.allclose(extension.kontext_latents, pack(expected), atol=1e-5)

    def test_dropping_the_shift_would_be_visible(self):
        """A guard on the guard: the shift is small, so a cell that cannot see it is not testing it."""
        vae = _tiny_vae()
        image = _reference_image()
        extension = _extension(vae, image)

        tensor = torch.from_numpy(torch.tensor(list(image.tobytes()), dtype=torch.uint8).numpy())
        tensor = tensor.reshape(64, 64, 3).permute(2, 0, 1).float() / 255.0
        tensor = (tensor * 2.0 - 1.0).unsqueeze(0)
        with torch.no_grad():
            raw = vae.encode(tensor).latent_dist.mode()
        without_shift = raw * vae.config.scaling_factor

        from invokeai.backend.flux.sampling_utils import pack

        assert not torch.allclose(extension.kontext_latents, pack(without_shift), atol=1e-5)


class TestTheOomFallback:
    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(torch.cuda.OutOfMemoryError("CUDA out of memory"), id="cuda"),
            pytest.param(
                RuntimeError("DefaultCPUAllocator: not enough memory: you tried to allocate 123456 bytes."),
                id="cpu-windows-android-mimalloc",
            ),
            pytest.param(
                RuntimeError(
                    "DefaultCPUAllocator: can't allocate memory: you tried to allocate "
                    "123456 bytes. Error code 12 (Cannot allocate memory)"
                ),
                id="cpu-posix",
            ),
        ],
    )
    def test_an_oom_retries_tiled_and_still_produces_the_reference(self, error):
        vae = _tiny_vae()
        image = _reference_image(edge=768)
        real_encode = vae.encode
        calls = {"n": 0}
        tiling_seen: list[bool] = []

        def fail_then_succeed(x, return_dict=True):
            calls["n"] += 1
            tiling_seen.append(vae.use_tiling)
            if calls["n"] == 1:
                raise error
            return real_encode(x, return_dict=return_dict)

        vae.encode = fail_then_succeed
        before = (vae.use_tiling, vae.tile_sample_min_size, vae.tile_latent_min_size)

        extension = _extension(vae, image)

        assert calls["n"] == 2
        assert tiling_seen == [False, True], "the retry is the tiled one"
        assert extension.kontext_latents.shape[-1] == 64, "a usable latent, not just a survived call"
        # The VAE is the cache's; the fallback must not leave it tiled for the next node.
        assert (vae.use_tiling, vae.tile_sample_min_size, vae.tile_latent_min_size) == before

    def test_a_non_oom_error_propagates(self):
        vae = _tiny_vae()

        def boom(x, return_dict=True):
            raise RuntimeError("Input type (float) and weight type (half) should be the same")

        vae.encode = boom

        with pytest.raises(RuntimeError, match="weight type"):
            _extension(vae, _reference_image())
