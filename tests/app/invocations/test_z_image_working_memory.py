"""The Z-Image VAE nodes reserve what they estimated.

Narrow on purpose: `test_z_image_tiled_decode.py` covers *what* is estimated (tile-bounded or not),
and this covers that the number reaches the model cache at all. Both nodes, because the reservation
is wired separately in each and either can be dropped without the other noticing.

The earlier version of this file wrapped each invocation in `except Exception: pass`, which made it
pass whether or not the node worked -- only the two mock assertions were load-bearing. The nodes run
for real here instead, against the tiny `AutoencoderKL` from `conftest`.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from invokeai.app.invocations.vae.z_image_image_to_latents import ZImageImageToLatentsInvocation
from invokeai.app.invocations.vae.z_image_latents_to_image import ZImageLatentsToImageInvocation


@pytest.fixture(autouse=True)
def _pin_to_cpu(monkeypatch):
    """Both nodes move their input to `choose_torch_device()`, which is CUDA on a machine that has
    one -- while the VAE here stays on the CPU, so the encode would die on a device mismatch. The
    previous version of this file hid exactly that behind `except Exception: pass`."""
    monkeypatch.setattr(
        "invokeai.backend.util.devices.TorchDevice.choose_torch_device", staticmethod(lambda: torch.device("cpu"))
    )


def _vae_info(vae) -> MagicMock:
    vae_info = MagicMock()
    vae_info.model = vae
    # Decode places latents on the VAE's intended compute device (see #9373); this must be a real
    # torch.device so `latents.to(device=...)` works instead of raising TypeError.
    vae_info.compute_device = torch.device("cpu")
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=(None, vae))
    cm.__exit__ = MagicMock(return_value=None)
    vae_info.model_on_device = MagicMock(return_value=cm)
    return vae_info


class TestTheEstimateReachesTheCache:
    def test_the_decode_node_reserves_what_it_estimated(self, flux_shaped_vae):
        vae = flux_shaped_vae()
        vae_info = _vae_info(vae)

        context = MagicMock()
        context.models.load.return_value = vae_info
        context.tensors.load.return_value = torch.zeros(1, 16, 64, 64)
        context.config.get.return_value.force_tiled_decode = False
        # `ImageOutput.build` validates these, so they cannot be bare MagicMocks.
        context.images.save.return_value = MagicMock(image_name="test.png", width=512, height=512)

        expected_memory = 500 * 1024 * 1024
        path = "invokeai.app.invocations.vae.z_image_latents_to_image.estimate_vae_working_memory_flux"
        with patch(path, return_value=expected_memory) as estimate:
            ZImageLatentsToImageInvocation.model_construct(
                latents=MagicMock(latents_name="test_latents"),
                vae=MagicMock(vae=MagicMock(), seamless_axes=[]),
                tiled=False,
                tile_size=0,
            ).invoke(context)

        estimate.assert_called_once()
        vae_info.model_on_device.assert_called_once_with(working_mem_bytes=expected_memory)

    def test_the_encode_node_reserves_what_it_estimated(self, flux_shaped_vae):
        vae = flux_shaped_vae()
        vae_info = _vae_info(vae)

        expected_memory = 250 * 1024 * 1024
        path = "invokeai.app.invocations.vae.z_image_image_to_latents.estimate_vae_working_memory_flux"
        with patch(path, return_value=expected_memory) as estimate:
            latents = ZImageImageToLatentsInvocation.vae_encode(vae_info, torch.zeros(1, 3, 512, 512))

        estimate.assert_called_once()
        vae_info.model_on_device.assert_called_once_with(working_mem_bytes=expected_memory)
        # The encode really ran: a 512px image through an 8x, 16-channel VAE is a 64px latent.
        assert latents.shape == (1, 16, 64, 64)
