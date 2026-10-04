"""Seeded noise is drawn in float32 unless `noise_dtype: float16` asks for the old half-precision draw.

float32 is what keeps a seed on the same image across torch versions: torch's float16/bfloat16 CPU `randn`/`rand`
changed after 2.7, its float32 ones did not. The setting tests compute their expectations from torch directly, with
float32/float64 targets where the two draws never coincide, so a draw site that stops honouring the setting fails on
any torch version. The golden values catch the other failure: a future torch changing the float32 draw itself.
"""

from unittest.mock import MagicMock

import pytest
import torch

from invokeai.app.invocations.cogview4.cogview4_denoise import CogView4DenoiseInvocation
from invokeai.app.invocations.fields import ZImageConditioningField
from invokeai.app.invocations.flux.flux_denoise import FluxDenoiseInvocation
from invokeai.app.invocations.flux2.flux2_denoise import Flux2DenoiseInvocation
from invokeai.app.invocations.latent_noise import generate_noise_tensor
from invokeai.app.invocations.sd3.sd3_denoise import SD3DenoiseInvocation
from invokeai.app.invocations.z_image.z_image_seed_variance_enhancer import ZImageSeedVarianceEnhancerInvocation
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    ConditioningFieldData,
    ZImageConditioningInfo,
)
from invokeai.backend.util import devices as devices_module

SEED = 1234
CPU = torch.device("cpu")


@pytest.fixture
def noise_dtype(monkeypatch: pytest.MonkeyPatch):
    def set_noise_dtype(value: str | None) -> None:
        """None leaves the setting at its default."""
        monkeypatch.delenv("INVOKEAI_NOISE_DTYPE", raising=False)
        config = InvokeAIAppConfig() if value is None else InvokeAIAppConfig(noise_dtype=value)
        monkeypatch.setattr(devices_module, "get_config", lambda: config)

    return set_noise_dtype


def seeded(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(*shape, dtype=dtype, generator=torch.Generator(device="cpu").manual_seed(SEED))


def prepared(invocation_class, *channels: int) -> torch.Tensor:
    """The noise a denoise node prepares when no noise is connected, cast to float32."""
    invocation = invocation_class.model_construct(width=64, height=64, seed=SEED, noise=None)
    return invocation._prepare_noise_tensor(MagicMock(), *channels, torch.float32, CPU)


# Each denoise node used to draw its own noise in float16 on the CPU.
DENOISE_NODES = {
    "flux": (lambda: prepared(FluxDenoiseInvocation), (1, 16, 8, 8)),
    "flux2": (lambda: prepared(Flux2DenoiseInvocation), (1, 32, 8, 8)),
    "sd3": (lambda: prepared(SD3DenoiseInvocation, 16), (1, 16, 8, 8)),
    "cogview4": (lambda: prepared(CogView4DenoiseInvocation, 16), (1, 16, 8, 8)),
}


def test_float32_draw_matches_the_values_every_supported_torch_gives(noise_dtype) -> None:
    # Measured on torch 2.7.1 (macOS pin) and 2.13.0 (Windows/Linux pins); a torch that changes these changes seeds.
    # The tolerance allows for the vectorized x86 and scalar/ARM `normal_fill` paths; an algorithm change moves O(1).
    noise_dtype(None)

    first = prepared(FluxDenoiseInvocation).flatten()[:8]

    expected = [-0.1117186, -0.4965901, 0.1630737, -0.8816878, 0.0539002, 0.6683737, -0.0596576, -0.4674979]
    assert torch.allclose(first, torch.tensor(expected), atol=1e-4)


@pytest.mark.parametrize("name", DENOISE_NODES)
def test_denoise_nodes_draw_in_float32_by_default(noise_dtype, name: str) -> None:
    noise_dtype(None)
    generate, shape = DENOISE_NODES[name]

    assert torch.equal(generate(), seeded(shape, torch.float32))


@pytest.mark.parametrize("name", DENOISE_NODES)
def test_float16_setting_restores_the_half_precision_draw(noise_dtype, name: str) -> None:
    noise_dtype("float16")
    generate, shape = DENOISE_NODES[name]

    noise = generate()

    assert torch.equal(noise, seeded(shape, torch.float16).float())
    assert not torch.equal(noise, seeded(shape, torch.float32))


@pytest.mark.parametrize("noise_type", ["SD", "FLUX", "FLUX.2", "SD3", "CogView4"])
def test_noise_node_draws_in_float32_and_keeps_the_device_precision(noise_dtype, noise_type: str) -> None:
    # float64 stands in for the device precision so the two draws differ on every torch version.
    noise_dtype(None)

    noise = generate_noise_tensor(noise_type, 64, 64, SEED, CPU, torch.float64)

    assert noise.dtype == torch.float64
    assert torch.equal(noise, seeded(tuple(noise.shape), torch.float32).double())


def test_noise_node_float16_setting_draws_in_the_device_precision(noise_dtype) -> None:
    noise_dtype("float16")

    noise = generate_noise_tensor("FLUX", 64, 64, SEED, CPU, torch.float64)

    assert torch.equal(noise, seeded((1, 16, 8, 8), torch.float64))


@pytest.mark.parametrize("setting", ["float32", "float16"])
@pytest.mark.parametrize(("noise_type", "shape"), [("Z-Image", (1, 16, 8, 8)), ("Anima", (1, 16, 1, 8, 8))])
def test_float32_noise_types_ignore_the_setting(noise_dtype, setting: str, noise_type: str, shape) -> None:
    noise_dtype(setting)

    noise = generate_noise_tensor(noise_type, 64, 64, SEED, CPU, torch.float16)

    assert torch.equal(noise, seeded(shape, torch.float32))


@pytest.mark.parametrize(("setting", "draw_dtype"), [(None, torch.float32), ("float16", torch.float64)])
def test_z_image_seed_variance_draws_in_the_configured_dtype(noise_dtype, setting, draw_dtype) -> None:
    """The enhancer's uniform draw follows the setting too; float64 embeddings stand in for the old bf16 ones."""
    noise_dtype(setting)
    embeds = torch.randn(4, 64, dtype=torch.float64, generator=torch.Generator().manual_seed(7))
    context = MagicMock()
    context.conditioning.load.return_value = ConditioningFieldData(
        conditionings=[ZImageConditioningInfo(prompt_embeds=embeds.clone())]
    )
    context.conditioning.save.return_value = "c-varied"
    invocation = ZImageSeedVarianceEnhancerInvocation(
        conditioning=ZImageConditioningField(conditioning_name="c"), seed=SEED, strength=0.5, randomize_percent=50.0
    )

    invocation.invoke(context)

    out = context.conditioning.save.call_args.args[0].conditionings[0].prompt_embeds
    changed = out != embeds
    uniform = torch.rand(embeds.shape, dtype=draw_dtype, generator=torch.Generator().manual_seed(SEED)).double()
    expected = (uniform * 2 - 1) * (0.5 * torch.std(embeds).item())
    assert changed.any()
    assert torch.allclose((out - embeds)[changed], expected[changed], atol=1e-12)
