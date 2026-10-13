"""How `SD3DenoiseInvocation` applies the LoRAs on its transformer, and which ones it refuses.

SD 3 Medium and Large name the same modules at different widths, and SD 3.5 Medium has dual attention (`attn2`) that
SD 3 Medium lacks. The patcher would only log and skip each layer that does not fit, so the node refuses such a LoRA
before the transformer moves. The fit checks run against the released geometry on the meta device; patching and the
working-memory request run a tiny transformer on the CPU, in float32.
"""

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from diffusers import SD3Transformer2DModel

from invokeai.app.invocations.model import LoRAField, ModelIdentifierField, TransformerField
from invokeai.app.invocations.sd3 import sd3_denoise
from invokeai.app.invocations.sd3.sd3_denoise import SD3DenoiseInvocation
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType, SD3VariantType
from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.lora_conversions.sd3_lora_constants import SD3_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw

_SD35_MEDIUM_CONFIG = {
    k: v
    for k, v in json.loads(
        (
            Path(__file__).parents[2]
            / "model_identification"
            / "stripped_models"
            / "5a5fe5e4-5776-43e9-a062-530080b2ead1"
            / "transformer"
            / "config.json"
        ).read_text()
    ).items()
    if not k.startswith("_")
}
_GIB = 1024**3


def _identifier(key: str, model_type: ModelType) -> ModelIdentifierField:
    return ModelIdentifierField(
        key=key, hash=f"h-{key}", name=key, base=BaseModelType.StableDiffusion3, type=model_type
    )


def _layer(rows: int, cols: int, generator: torch.Generator | None = None) -> LoRALayer:
    if generator is None:
        return LoRALayer(up=torch.zeros(rows, 4), mid=None, down=torch.zeros(4, cols), alpha=None, bias=None)
    up = torch.randn(rows, 4, generator=generator) * 0.1
    return LoRALayer(up=up, mid=None, down=torch.randn(4, cols, generator=generator), alpha=None, bias=None)


def _patch(layers: dict[str, LoRALayer]) -> ModelPatchRaw:
    return ModelPatchRaw(layers={f"{SD3_LORA_TRANSFORMER_PREFIX}{path}": layer for path, layer in layers.items()})


def _fit_check(transformer: SD3Transformer2DModel, patch: ModelPatchRaw, lora_variant: SD3VariantType | None) -> None:
    lora = LoRAField(lora=_identifier("lora", ModelType.LoRA), weight=1.0)
    invocation = SD3DenoiseInvocation.model_construct(
        transformer=TransformerField(transformer=_identifier("main", ModelType.Main), loras=[lora])
    )
    context = SimpleNamespace(
        models=SimpleNamespace(get_config=lambda _identifier: SimpleNamespace(variant=lora_variant))
    )
    invocation._raise_unless_loras_fit(context, transformer, [(patch, 1.0)])


@pytest.fixture(scope="module")
def sd35_medium() -> SD3Transformer2DModel:
    with torch.device("meta"):
        return SD3Transformer2DModel(**_SD35_MEDIUM_CONFIG)


@pytest.fixture(scope="module")
def sd3_medium() -> SD3Transformer2DModel:
    """SD 3 Medium: the same width as 3.5 Medium, without its dual attention."""
    with torch.device("meta"):
        return SD3Transformer2DModel(**{**_SD35_MEDIUM_CONFIG, "dual_attention_layers": ()})


def test_a_lora_that_fits_passes(sd35_medium: SD3Transformer2DModel) -> None:
    patch = _patch({"transformer_blocks.0.attn2.to_q": _layer(1536, 1536), "pos_embed.proj": _layer(1536, 16)})
    patch.layers[f"{SD3_LORA_TRANSFORMER_PREFIX}pos_embed.proj"] = LoRALayer(
        up=torch.zeros(1536, 4, 1, 1), mid=None, down=torch.zeros(4, 16, 2, 2), alpha=None, bias=None
    )

    _fit_check(sd35_medium, patch, SD3VariantType.Medium)


def test_a_large_lora_is_refused_on_a_medium_model(sd35_medium: SD3Transformer2DModel) -> None:
    patch = _patch({"transformer_blocks.0.attn.to_q": _layer(2432, 2432)})

    with pytest.raises(ValueError, match="trained for SD 3.5 Large and does not fit this SD 3 / 3.5 Medium model"):
        _fit_check(sd35_medium, patch, SD3VariantType.Large)


def test_a_lora_without_a_recorded_size_is_refused_by_its_shapes(sd35_medium: SD3Transformer2DModel) -> None:
    """Records installed before the variant existed carry none; the shapes still decide."""
    patch = _patch({"transformer_blocks.0.attn.to_q": _layer(2432, 2432)})

    with pytest.raises(ValueError, match="does not fit this SD 3 model.*another SD 3 size"):
        _fit_check(sd35_medium, patch, None)


def test_a_35_medium_lora_with_dual_attention_is_refused_on_sd3_medium(sd3_medium: SD3Transformer2DModel) -> None:
    patch = _patch({"transformer_blocks.0.attn2.to_q": _layer(1536, 1536)})

    with pytest.raises(ValueError, match="dual attention"):
        _fit_check(sd3_medium, patch, SD3VariantType.Medium)


class _StopBeforeDenoising(Exception):
    """Raised where the loop starts, with the LoRAs applied."""


_TINY = {
    "sample_size": 8,
    "patch_size": 2,
    "in_channels": 4,
    "num_layers": 2,
    "attention_head_dim": 8,
    "num_attention_heads": 2,
    "joint_attention_dim": 8,
    "caption_projection_dim": 16,
    "pooled_projection_dim": 8,
    "out_channels": 4,
    "pos_embed_max_size": 8,
    "dual_attention_layers": (0,),
}
_QKV = "transformer_blocks.0.attn.to_q"


def _run(
    monkeypatch,
    model: SD3Transformer2DModel,
    model_format: ModelFormat,
    patch: ModelPatchRaw,
    while_patched,
    requested: list | None = None,
):
    """Run the node up to its loop with a mocked cache; return the working memory it asked the cache for.

    `requested` collects every lock taken on the transformer, also when the node raises before its loop.
    """
    from invokeai.backend.util.devices import TorchDevice

    # A CPU bfloat16 matmul aborts some CI runners, and low precision is not what is under test.
    monkeypatch.setattr(TorchDevice, "choose_torch_dtype", staticmethod(lambda *args, **kwargs: torch.float32))
    monkeypatch.setattr(TorchDevice, "choose_torch_device", staticmethod(lambda *args, **kwargs: torch.device("cpu")))
    monkeypatch.setattr(
        SD3DenoiseInvocation,
        "_load_text_conditioning",
        lambda self, **kwargs: (torch.zeros(1, 3, 8), torch.zeros(1, 8)),
    )

    def tqdm(iterable, **kwargs):
        while_patched(model)
        raise _StopBeforeDenoising

    monkeypatch.setattr(sd3_denoise, "tqdm", tqdm)
    requested = [] if requested is None else requested

    @contextmanager
    def on_device(working_mem_bytes=None):
        requested.append(working_mem_bytes)
        yield None, model

    transformer_info = SimpleNamespace(model=model, model_on_device=on_device)
    lora_info = SimpleNamespace(model=patch, model_in_ram=lambda: contextmanager(lambda: (yield patch))())
    context = MagicMock()
    context.models.load.side_effect = lambda identifier: transformer_info if identifier.key == "main" else lora_info
    context.models.get_config.side_effect = lambda identifier: SimpleNamespace(format=model_format, variant=None)
    context.config.get.return_value = SimpleNamespace(device_working_mem_gb=3.0)
    lora = LoRAField(lora=_identifier("lora", ModelType.LoRA), weight=1.0)
    invocation = SD3DenoiseInvocation.model_construct(
        transformer=TransformerField(transformer=_identifier("main", ModelType.Main), loras=[lora]),
        positive_conditioning=SimpleNamespace(conditioning_name="pos"),
        negative_conditioning=SimpleNamespace(conditioning_name="neg"),
        cfg_scale=1.0,
        steps=2,
        width=16,
        height=16,
        seed=0,
        denoising_start=0.0,
        denoising_end=1.0,
        latents=None,
        noise=None,
        denoise_mask=None,
    )

    with pytest.raises(_StopBeforeDenoising):
        invocation.invoke(context)
    return requested[0]


@pytest.mark.parametrize("fp8", [False, True], ids=["patched-in-place", "fp8-sidecar"])
def test_the_node_applies_its_loras_and_reserves_what_stays_on_the_device(monkeypatch, fp8: bool) -> None:
    """A plain transformer takes the LoRA into its weights and needs no room for it; FP8 Storage weights have no add
    kernel, so the patcher puts the LoRA on the sidecar, which stays on the device on top of the working-memory floor.
    Both must compute with the LoRA while denoising."""
    torch.manual_seed(0)
    model = SD3Transformer2DModel(**_TINY)
    if fp8:
        to_q = model.get_submodule(_QKV)
        to_q.weight = torch.nn.Parameter(to_q.weight.detach().to(torch.float8_e4m3fn), requires_grad=False)
    apply_custom_layers_to_model(model)
    weight = model.get_submodule(_QKV).weight.detach().float().clone()
    generator = torch.Generator().manual_seed(1)
    layer = _layer(16, 16, generator)
    patch = _patch({_QKV: layer})
    x = torch.randn(2, 16, generator=generator)
    seen: list[torch.Tensor] = []

    requested = _run(monkeypatch, model, ModelFormat.Diffusers, patch, lambda m: seen.append(m.get_submodule(_QKV)(x)))

    torch.testing.assert_close(seen[0], x @ (weight + layer.up @ layer.down).T + model.get_submodule(_QKV).bias.float())
    if fp8:
        assert requested == 3 * _GIB + patch.calc_size() + _GIB
    else:
        assert requested is None


def test_the_node_refuses_a_lora_that_does_not_fit_before_it_locks_the_transformer(monkeypatch) -> None:
    model = SD3Transformer2DModel(**_TINY)
    patch = _patch({_QKV: _layer(32, 32)})  # another size's width
    locks: list = []

    with pytest.raises(ValueError, match="does not fit"):
        _run(monkeypatch, model, ModelFormat.Diffusers, patch, lambda _m: None, locks)
    # Refused before the transformer moved: nothing was evicted for a generation that cannot run.
    assert locks == []
