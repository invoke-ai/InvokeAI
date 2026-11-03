"""The vendored MOSS-Image transformer on a device without float64.

The reference computes its timestep sinusoid in float64 on the latents' device, which MPS cannot do: every MOSS-Image
generation on an Apple GPU failed at the first forward. The local change computes it on the CPU there. It must give
what the on-device path gives, bit for bit, and on MPS it must run at all.
"""

import pytest
import torch

from invokeai.backend.moss_image import pipeline_moss_image
from invokeai.backend.moss_image.pipeline_moss_image import MOSSImageTransformer


def _transformer() -> MOSSImageTransformer:
    torch.manual_seed(0)
    return MOSSImageTransformer(
        dim=64, ffn_dim=96, num_heads=2, num_layers=2, text_dim=24, freq_dim=256, latent_channels=8
    ).eval()


def _inputs(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    latents = torch.randn(1, 8, 4, 6, generator=generator)
    text = torch.randn(1, 5, 24, generator=generator)
    # Late in the schedule, where a float32 sinusoid would visibly differ: phases up to ~1000 radians.
    timestep = torch.tensor(987.0)
    return latents.to(device), timestep.to(device), text.to(device)


@torch.no_grad()
def test_the_cpu_route_for_the_sinusoid_gives_the_on_device_result_bit_for_bit(monkeypatch: pytest.MonkeyPatch) -> None:
    transformer = _transformer()
    inputs = _inputs(torch.device("cpu"))
    expected = transformer(*inputs).sample

    monkeypatch.setattr(pipeline_moss_image, "_supports_float64", lambda _device: False)

    assert torch.equal(transformer(*inputs).sample, expected)


def test_only_mps_takes_the_cpu_route() -> None:
    assert not pipeline_moss_image._supports_float64(torch.device("mps"))
    for device in ("cpu", "cuda", "xpu"):
        assert pipeline_moss_image._supports_float64(torch.device(device))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an Apple GPU (MPS)")
@torch.no_grad()
def test_a_forward_runs_on_mps_and_matches_the_cpu() -> None:
    transformer = _transformer()
    expected = transformer(*_inputs(torch.device("cpu"))).sample

    mps = torch.device("mps")
    result = transformer.to(mps)(*_inputs(mps)).sample

    # Float32 on both; MPS kernels round differently from the CPU's.
    torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)
