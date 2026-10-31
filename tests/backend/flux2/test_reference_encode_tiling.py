"""Which reference encode `encode_reference_image` runs: tiled always (FLUX.2), or untiled where it can (FLUX.3).

A tiled encode is a different encode, not a cheaper copy of the untiled one: each tile's mid-block attention
sees only that tile. These tests pin which one each mode produces, on a tiny real FLUX.2 VAE, and that the
shared VAE's tiling settings survive the call.
"""

from contextlib import contextmanager

import pytest
import torch
import torchvision.transforms as T
from diffusers import AutoencoderKLFlux2
from PIL import Image

import invokeai.backend.flux2.ref_image_extension as ref_image_extension
from invokeai.backend.flux2.ref_image_extension import REF_ENCODE_TILE_SIZE, encode_reference_image
from invokeai.backend.working_memory.estimate import WorkingMemoryEstimate, WorkingMemoryTerm

# Larger than one 512 px tile, so the tiled and untiled encodes differ.
SIZE = 768


class _LoadedVae:
    """The part of `LoadedModel` the encode uses."""

    def __init__(self, vae: AutoencoderKLFlux2):
        self.model = vae

    @contextmanager
    def model_on_device(self, *, working_memory: WorkingMemoryEstimate):
        yield None, self.model


@pytest.fixture(scope="module")
def vae() -> AutoencoderKLFlux2:
    torch.manual_seed(0)
    return AutoencoderKLFlux2(
        block_out_channels=(32, 64, 64, 64), latent_channels=32, layers_per_block=1, norm_num_groups=32
    ).eval()


@pytest.fixture(scope="module")
def image() -> Image.Image:
    pixels = torch.rand(3, SIZE, SIZE, generator=torch.Generator().manual_seed(1))
    return T.ToPILImage()(pixels)


@pytest.fixture(autouse=True)
def _cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ref_image_extension.TorchDevice, "choose_torch_device", lambda: torch.device("cpu"))


def _direct_encode(vae: AutoencoderKLFlux2, image: Image.Image, tiled: bool) -> torch.Tensor:
    pixels = (T.ToTensor()(image) * 2.0 - 1.0).unsqueeze(0)
    saved = (vae.use_tiling, vae.tile_sample_min_size, vae.tile_latent_min_size)
    vae.use_tiling = tiled
    vae.tile_sample_min_size, vae.tile_latent_min_size = REF_ENCODE_TILE_SIZE, REF_ENCODE_TILE_SIZE // 8
    try:
        with torch.no_grad():
            return vae.encode(pixels, return_dict=False)[0].mode()
    finally:
        vae.use_tiling, vae.tile_sample_min_size, vae.tile_latent_min_size = saved


@pytest.fixture(scope="module")
def expected(vae: AutoencoderKLFlux2, image: Image.Image) -> dict[str, torch.Tensor]:
    encodes = {"tiled": _direct_encode(vae, image, tiled=True), "untiled": _direct_encode(vae, image, tiled=False)}
    assert not torch.allclose(encodes["tiled"], encodes["untiled"], atol=1e-4), "precondition: tiling matters here"
    return encodes


def _encode(vae: AutoencoderKLFlux2, image: Image.Image, **kwargs) -> torch.Tensor:
    with torch.no_grad():
        return encode_reference_image(_LoadedVae(vae), image, **kwargs)


def test_flux2_keeps_its_tiled_encode(vae, image, expected) -> None:
    assert torch.allclose(_encode(vae, image), expected["tiled"], atol=1e-5)


@pytest.mark.parametrize(("materializes", "kind"), [(False, "untiled"), (True, "tiled")])
def test_the_conditional_mode_tiles_only_where_attention_would_materialize(
    vae, image, expected, monkeypatch: pytest.MonkeyPatch, materializes: bool, kind: str
) -> None:
    monkeypatch.setattr(ref_image_extension, "flux2_vae_attention_materializes", lambda *_: materializes)

    latents = _encode(vae, image, tiling="if_attention_materializes")

    assert torch.allclose(latents, expected[kind], atol=1e-5)


def test_the_shared_vae_keeps_its_own_tiling_settings(vae, image, monkeypatch: pytest.MonkeyPatch) -> None:
    """The VAE is a cached instance; the final decode must not inherit the reference encode's settings."""
    before = (vae.use_tiling, vae.tile_sample_min_size, vae.tile_latent_min_size)
    monkeypatch.setattr(ref_image_extension, "flux2_vae_attention_materializes", lambda *_: False)

    _encode(vae, image, tiling="if_attention_materializes")
    _encode(vae, image)

    assert (vae.use_tiling, vae.tile_sample_min_size, vae.tile_latent_min_size) == before


@pytest.mark.parametrize(
    ("total_gb", "estimate_gb", "expected"),
    [(24, 9.2, None), (8, 9.2, REF_ENCODE_TILE_SIZE), (8, 2.3, None)],
    ids=["4mp-on-24gb", "4mp-on-8gb", "1mp-on-8gb"],
)
def test_an_untiled_encode_is_taken_only_where_it_fits_the_card(
    vae, monkeypatch: pytest.MonkeyPatch, total_gb: int, estimate_gb: float, expected: int | None
) -> None:
    """Fused attention keeps an untiled encode linear in area, but a 4 MP one still outgrows an 8 GB card."""
    monkeypatch.setattr(ref_image_extension, "flux2_vae_attention_materializes", lambda *_: False)
    estimate = WorkingMemoryEstimate(
        estimator="vae.flux2",
        version=1,
        features={},
        terms=(WorkingMemoryTerm("untiled encode", estimate_gb * 2**30, True),),
        source="test",
    )
    monkeypatch.setattr(ref_image_extension, "estimate_vae_working_memory_flux2", lambda **_: estimate)
    monkeypatch.setattr(
        ref_image_extension.torch.cuda,
        "get_device_properties",
        lambda device: type("Props", (), {"total_memory": total_gb * 2**30})(),
    )
    pixels = torch.zeros(1, 3, 64, 64)

    assert (
        ref_image_extension.reference_encode_tile_size(
            pixels, vae, torch.device("cuda"), tiling="if_attention_materializes"
        )
        == expected
    )
    assert (
        ref_image_extension.reference_encode_tile_size(pixels, vae, torch.device("cuda"), tiling="always")
        == REF_ENCODE_TILE_SIZE
    )
