"""The FLUX.1 and Z-Image VAE nodes feed an SDNQ VAE in the dtype it computes in.

An SDNQ VAE can lead with a packed weight, an `SDNQTensor` whose `dtype` is its integer storage. An image cast to
that dtype reaches the convolution as an integer tensor, and the run fails with "Input type (signed char) and bias type
(float) should be the same". The nodes read the dtype the way the working-memory estimator does (`vae_compute_dtype`).

The VAE is a real tiny `AutoencoderKL` whose first weight is replaced by a packed one. Its `encode` and `decode` are
recorded rather than run: a bf16 forward on the CPU is not what is under test, the dtype each node hands the VAE is.
"""

from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.app.invocations.vae.cogview4_image_to_latents import CogView4ImageToLatentsInvocation
from invokeai.app.invocations.vae.flux_vae_decode import FluxVaeDecodeInvocation
from invokeai.app.invocations.vae.flux_vae_encode import FluxVaeEncodeInvocation
from invokeai.app.invocations.vae.sd3_image_to_latents import SD3ImageToLatentsInvocation
from invokeai.app.invocations.vae.z_image_image_to_latents import ZImageImageToLatentsInvocation
from invokeai.app.invocations.vae.z_image_latents_to_image import ZImageLatentsToImageInvocation
from invokeai.backend.quantization.sdnq.sdnq_tensor import SDNQTensor
from invokeai.backend.quantization.sdnq.utils import SDNQQuantizationType

COMPUTE_DTYPE = torch.bfloat16


@pytest.fixture(autouse=True)
def _pin_to_cpu(monkeypatch):
    monkeypatch.setattr(
        "invokeai.backend.util.devices.TorchDevice.choose_torch_device", staticmethod(lambda: torch.device("cpu"))
    )


def _sdnq_vae(flux_shaped_vae):
    """A FLUX-shaped VAE whose first parameter is a packed int8 weight that dequantizes to bf16; `seen` records the
    dtype of what `encode` and `decode` are called with."""
    vae = flux_shaped_vae()
    weight = vae.encoder.conv_in.weight
    out_features = weight.shape[0]
    packed = SDNQTensor(
        data=torch.zeros(out_features, weight[0].numel(), dtype=torch.int8),
        quantization_type=SDNQQuantizationType.INT8_SYM,
        tensor_shape=weight.shape,
        compute_dtype=COMPUTE_DTYPE,
        scale=torch.ones(out_features, 1, dtype=torch.float32),
    )
    vae.encoder.conv_in.weight = torch.nn.Parameter(packed, requires_grad=False)
    first = next(vae.parameters())
    assert isinstance(first, SDNQTensor) and not first.is_floating_point(), "the fixture must lead with packed weights"

    seen: dict[str, torch.dtype] = {}

    def encode(x, return_dict=True):
        seen["encode"] = x.dtype
        dist = MagicMock()
        dist.sample.return_value = dist.mode.return_value = torch.zeros(1, 16, 8, 8)
        return MagicMock(latent_dist=dist)

    def decode(z, return_dict=True):
        seen["decode"] = z.dtype
        return (torch.zeros(1, 3, 64, 64),)

    vae.encode, vae.decode = encode, decode
    return vae, seen


def _loaded(vae) -> MagicMock:
    loaded = MagicMock()
    loaded.model = vae
    loaded.compute_device = torch.device("cpu")

    @contextmanager
    def on_device(working_memory=None):
        yield (None, vae)

    loaded.model_on_device = on_device
    return loaded


def _context() -> MagicMock:
    context = MagicMock()
    context.config.get.return_value.force_tiled_decode = False
    context.config.get.return_value.auto_tiled_decode = False
    return context


def test_the_flux_encode_casts_the_image_to_the_compute_dtype_of_an_sdnq_vae(flux_shaped_vae) -> None:
    vae, seen = _sdnq_vae(flux_shaped_vae)

    FluxVaeEncodeInvocation.vae_encode(
        vae_info=_loaded(vae), image_tensor=torch.zeros(1, 3, 64, 64), tiled=False, tile_size=0
    )

    assert seen["encode"] is COMPUTE_DTYPE


def test_the_z_image_encode_casts_the_image_and_returns_the_latents_in_the_compute_dtype(flux_shaped_vae) -> None:
    vae, seen = _sdnq_vae(flux_shaped_vae)

    latents = ZImageImageToLatentsInvocation.vae_encode(
        vae_info=_loaded(vae), image_tensor=torch.zeros(1, 3, 64, 64), tiled=False, tile_size=0
    )

    assert seen["encode"] is COMPUTE_DTYPE
    assert latents.dtype is COMPUTE_DTYPE


def test_the_flux_decode_casts_the_latents_to_the_compute_dtype_of_an_sdnq_vae(flux_shaped_vae) -> None:
    vae, seen = _sdnq_vae(flux_shaped_vae)

    FluxVaeDecodeInvocation.model_construct()._vae_decode(
        context=_context(), vae_info=_loaded(vae), latents=torch.zeros(1, 16, 8, 8)
    )

    assert seen["decode"] is COMPUTE_DTYPE


def test_the_z_image_decode_casts_the_latents_to_the_compute_dtype_of_an_sdnq_vae(flux_shaped_vae) -> None:
    vae, seen = _sdnq_vae(flux_shaped_vae)
    context = _context()
    context.tensors.load.return_value = torch.zeros(1, 16, 8, 8)
    context.models.load.return_value = _loaded(vae)
    context.images.save.return_value = MagicMock(image_name="out.png", width=64, height=64)
    node = ZImageLatentsToImageInvocation.model_construct(
        latents=MagicMock(latents_name="latents"),
        vae=MagicMock(vae=MagicMock(), seamless_axes=[]),
        tiled=False,
        tile_size=0,
    )

    node.invoke(context)

    assert seen["decode"] is COMPUTE_DTYPE


@pytest.mark.parametrize("node", [SD3ImageToLatentsInvocation, CogView4ImageToLatentsInvocation])
def test_the_sd3_and_cogview4_encodes_cast_to_the_compute_dtype_not_the_first_float_parameter(
    flux_shaped_vae, node
) -> None:
    """diffusers' `vae.dtype` skips the packed weight and reads the first floating-point one, here an fp32 stored
    parameter, which is not what the packed layers dequantize to."""
    vae, seen = _sdnq_vae(flux_shaped_vae)
    assert vae.dtype is torch.float32

    latents = node.vae_encode(vae_info=_loaded(vae), image_tensor=torch.zeros(1, 3, 64, 64))

    assert seen["encode"] is COMPUTE_DTYPE
    assert latents.dtype is COMPUTE_DTYPE
