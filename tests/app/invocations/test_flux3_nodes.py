"""The FLUX.3 LoRA nodes and the LoRA path of `Flux3DenoiseInvocation`, against stand-ins for the model and the context.

The conversion and the patcher run on a real transformer in `test_flux3_lora_conversion_utils.py`. What only the nodes
decide is tested here: which LoRAs reach the transformer, merged into its weights or beside them as sidecars by its
format, for the whole sampling and not after it, and what the working-memory reservation adds for them.
"""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

import invokeai.app.invocations.flux3.flux3_denoise as flux3_denoise
from invokeai.app.invocations.flux3.flux3_denoise import Flux3DenoiseInvocation
from invokeai.app.invocations.flux3.flux3_lora_loader import Flux3LoRACollectionLoader, Flux3LoRALoaderInvocation
from invokeai.app.invocations.model import LoRAField, ModelIdentifierField, TransformerField
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.model_manager.taxonomy import BaseModelType, Flux3VariantType, ModelFormat, ModelType
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.lora_conversions.flux3_lora_conversion_utils import lora_model_from_flux3_state_dict
from invokeai.backend.quantization.int8_convrot import Int8ConvrotLinear
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningFieldData, Flux3ConditioningInfo

_LORA_NAME = "Angles"


def _lora_context(base: BaseModelType, model_type: ModelType = ModelType.LoRA):
    configs = {"lora": SimpleNamespace(name=_LORA_NAME, base=base, type=model_type)}
    return SimpleNamespace(
        models=SimpleNamespace(exists=lambda key: key in configs, get_config=lambda key: configs[key])
    )


def _lora_field(weight: float = 0.8, key: str = "lora") -> LoRAField:
    return LoRAField(
        lora=ModelIdentifierField(key=key, hash="h", name=_LORA_NAME, base=BaseModelType.Flux3, type=ModelType.LoRA),
        weight=weight,
    )


def _transformer_field() -> TransformerField:
    model = ModelIdentifierField(key="transformer", hash="h", name="m", base=BaseModelType.Flux3, type=ModelType.Main)
    return TransformerField(transformer=model, loras=[])


def _lora_loader(node_class: str, lora: LoRAField):
    if node_class == "single":
        return Flux3LoRALoaderInvocation.model_construct(
            lora=lora.lora, weight=lora.weight, transformer=_transformer_field()
        )
    return Flux3LoRACollectionLoader.model_construct(loras=[lora], transformer=_transformer_field())


def test_the_loader_applies_a_lora_at_its_weight() -> None:
    output = _lora_loader("single", _lora_field(0.6)).invoke(_lora_context(BaseModelType.Flux3))

    assert output.transformer is not None
    assert [(lora.lora.key, lora.weight) for lora in output.transformer.loras] == [("lora", 0.6)]


def test_a_lora_collection_adds_each_lora_once() -> None:
    """Including one an upstream loader applied: a LoRA applied twice doubles its weight."""
    applied = _transformer_field()
    applied.loras.append(_lora_field(0.3, key="upstream"))
    configs = {
        key: SimpleNamespace(name=key, base=BaseModelType.Flux3, type=ModelType.LoRA) for key in ("lora", "upstream")
    }
    context = SimpleNamespace(models=SimpleNamespace(exists=configs.__contains__, get_config=configs.__getitem__))

    output = Flux3LoRACollectionLoader.model_construct(
        loras=[_lora_field(0.8), _lora_field(0.5), _lora_field(1.0, key="upstream")], transformer=applied
    ).invoke(context)

    assert output.transformer is not None
    assert [(lora.lora.key, lora.weight) for lora in output.transformer.loras] == [("upstream", 0.3), ("lora", 0.8)]


def test_the_loader_refuses_a_lora_already_on_the_transformer() -> None:
    applied = _transformer_field()
    applied.loras.append(_lora_field(0.3))
    node = Flux3LoRALoaderInvocation.model_construct(lora=_lora_field().lora, weight=1.0, transformer=applied)

    with pytest.raises(ValueError, match="already applied"):
        node.invoke(_lora_context(BaseModelType.Flux3))


@pytest.mark.parametrize("node_class", ["single", "collection"])
@pytest.mark.parametrize("base", [BaseModelType.Flux, BaseModelType.Flux2])
def test_another_familys_lora_is_refused(node_class: str, base: BaseModelType) -> None:
    """FLUX.1 and FLUX.2 LoRAs share module names with FLUX.3, but not its layers."""
    with pytest.raises(ValueError, match="not FLUX.3"):
        _lora_loader(node_class, _lora_field()).invoke(_lora_context(base))


@pytest.mark.parametrize("node_class", ["single", "collection"])
def test_a_lora_that_is_not_installed_is_refused(node_class: str) -> None:
    with pytest.raises(ValueError, match="Unknown LoRA: gone"):
        _lora_loader(node_class, _lora_field(key="gone")).invoke(_lora_context(BaseModelType.Flux3))


@pytest.mark.parametrize("node_class", ["single", "collection"])
def test_a_model_that_is_not_a_lora_is_refused(node_class: str) -> None:
    with pytest.raises(ValueError, match="not a LoRA"):
        _lora_loader(node_class, _lora_field()).invoke(_lora_context(BaseModelType.Flux3, ModelType.Main))


class _Transformer(torch.nn.Module):
    """The released geometry's config, which the reservation prices on, and Linears a LoRA can target: the output
    projection and modulation."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(flow_map=True, num_attention_heads=24, attention_head_dim=128)
        self.proj_out = torch.nn.Linear(8, 4, bias=False)
        self.norm_out = torch.nn.Module()
        self.norm_out.linear = torch.nn.Linear(8, 4, bias=False)


class _TransformerInfo:
    """Records the working memory the transformer is locked with."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.reserved: list = []

    @contextmanager
    def model_on_device(self, *, working_memory):
        self.reserved.append(working_memory)
        yield ({}, self.model)


_BASE = torch.full((4, 8), 0.125)
_UP, _DOWN = torch.full((4, 2), 0.5), torch.full((2, 8), 0.25)
_SIZE = 128
# One sample's joint sequence: 8 x 8 image tokens at 128 px, and the caption's 80.
_TOKENS = (_SIZE // 16) ** 2 + 80


def _lora(module: str = "proj_out", **values: torch.Tensor):
    """A LoRA on one module, as the conversion builds it from a diffusers/PEFT file."""
    values = values or {"lora_A.weight": _DOWN, "lora_B.weight": _UP}
    state_dict = {f"transformer.{module}.{name}": value for name, value in values.items()}
    return lora_model_from_flux3_state_dict(state_dict, name=_LORA_NAME)


def _run(
    transformer_info: _TransformerInfo,
    model_format: ModelFormat,
    loras: dict[str, tuple] | None = None,
    lora_base: BaseModelType = BaseModelType.Flux3,
) -> dict:
    """`_run_diffusion` with `loras` ({key: (patch, weight)}, installed under `lora_base`) on the transformer, `denoise`
    replaced by a forward of its output projection. Returns what that forward saw and whether the projection's weight
    was patched then."""
    loras = loras or {}
    weight_before = transformer_info.model.proj_out.weight.clone()
    vae_info = MagicMock()
    vae_info.model_on_device.return_value.__enter__.return_value = (None, MagicMock())
    infos = {"vae": vae_info, "transformer": transformer_info}
    infos |= {key: SimpleNamespace(model=lora, model_in_ram=nullcontext) for key, (lora, _) in loras.items()}
    configs = {
        "transformer": SimpleNamespace(
            base=BaseModelType.Flux3,
            type=ModelType.Main,
            name="flux3",
            variant=Flux3VariantType.Klein,
            format=model_format,
        ),
        **{key: SimpleNamespace(base=lora_base, type=ModelType.LoRA, name=_LORA_NAME) for key in loras},
    }

    context = MagicMock()
    context.models.get_config.side_effect = lambda identifier: configs[identifier.key]
    context.models.load.side_effect = lambda identifier: infos[getattr(identifier, "key", identifier)]
    context.conditioning.load.return_value = ConditioningFieldData(
        conditionings=[Flux3ConditioningInfo(prompt_embeds=torch.zeros(1, 80, 8))]
    )
    transformer_field = _transformer_field()
    transformer_field.loras.extend(_lora_field(weight, key=key) for key, (_, weight) in loras.items())
    node = Flux3DenoiseInvocation.model_construct(
        latents=None,
        denoise_mask=None,
        denoising_start=0.0,
        denoising_end=1.0,
        add_noise=True,
        transformer=transformer_field,
        positive_conditioning=SimpleNamespace(conditioning_name="pos"),
        negative_conditioning=None,
        cfg_scale=1.0,
        width=_SIZE,
        height=_SIZE,
        steps=2,
        seed=0,
        vae=SimpleNamespace(vae="vae"),
        kontext_conditioning=None,
    )
    seen: dict = {}

    def denoise(**kwargs):
        model = kwargs["model"]
        seen["out"] = model.proj_out(torch.ones(1, 8))
        seen["patched"] = not torch.equal(model.proj_out.weight, weight_before)
        return kwargs["img"]

    with (
        patch.object(flux3_denoise.TorchDevice, "choose_torch_device", return_value=torch.device("cpu")),
        # float32: CPU bf16 matmuls fault on part of the hosted Windows fleet.
        patch.object(flux3_denoise.TorchDevice, "choose_bfloat16_safe_dtype", return_value=torch.float32),
        patch.object(flux3_denoise, "read_vae_bn_stats", return_value=(torch.zeros(128), torch.ones(128))),
        patch.object(flux3_denoise, "denoise", denoise),
    ):
        node._run_diffusion(context)
    return seen


def _transformer() -> _Transformer:
    transformer = _Transformer()
    with torch.no_grad():
        transformer.proj_out.weight.copy_(_BASE)
    # As the model cache holds it, so the patcher can wrap a layer in a sidecar.
    apply_custom_layers_to_model(transformer)
    return transformer


def _int8_transformer() -> _Transformer:
    transformer = _Transformer()
    transformer.proj_out = Int8ConvrotLinear(torch.ones(4, 8, dtype=torch.int8), torch.full((4, 1), 0.125), False)
    apply_custom_layers_to_model(transformer)
    return transformer


@pytest.mark.parametrize(
    ("make", "model_format", "merged"),
    [
        (_transformer, ModelFormat.Checkpoint, True),
        (_transformer, ModelFormat.GGUFQuantized, False),
        (_int8_transformer, ModelFormat.Checkpoint, False),
    ],
    ids=["bf16-merged", "gguf-sidecar", "int8-sidecar"],
)
def test_loras_reach_the_transformer_for_the_sampling_and_leave_it_unchanged(
    make, model_format: ModelFormat, merged: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A GGUF weight is packed and an int8 one is kept in buffers: neither takes an update in place, so their LoRAs run
    as sidecars, the int8 transformer's by its module tree, whatever its format says."""
    # As if the layer were on the GPU: on the CPU the patcher runs every LoRA as a sidecar, whatever the format.
    monkeypatch.setattr(LayerPatcher, "_is_any_part_of_layer_on_cpu", staticmethod(lambda _module: False))
    transformer = make()

    seen = _run(_TransformerInfo(transformer), model_format, {"lora": (_lora(), 0.5)})

    # The weight 0.125 everywhere, plus the LoRA at weight 0.5 (scale 1: no alpha), on an input of ones.
    expected = torch.full((4,), 8 * 0.125) + (_UP @ _DOWN * 0.5) @ torch.ones(8)
    torch.testing.assert_close(seen["out"][0], expected)
    assert seen["patched"] is merged
    torch.testing.assert_close(transformer.proj_out(torch.ones(1, 8))[0], torch.full((4,), 1.0))


@pytest.mark.parametrize("base", [BaseModelType.Flux, BaseModelType.Flux2])
def test_another_familys_lora_is_refused_before_anything_loads(base: BaseModelType) -> None:
    """Another family's loader node wired into a FLUX.3 graph hands its LoRA over unchecked; converted for that
    family's transformer, its layers match none of this one's."""
    transformer_info = _TransformerInfo(_transformer())

    with pytest.raises(ValueError, match=f"LoRA '{_LORA_NAME}' is for {base.value} models, not FLUX.3"):
        _run(transformer_info, ModelFormat.Checkpoint, {"lora": (_lora(), 1.0)}, lora_base=base)
    assert transformer_info.reserved == []


def test_the_reservation_prices_the_loras_the_sampling_applies() -> None:
    """The estimate prices them (`denoise_working_memory`); the node hands it those at a weight other than 0, with the
    transformer's format: on a GGUF transformer every layer runs as a sidecar."""
    without, at_zero, applied = (_TransformerInfo(_transformer()) for _ in range(3))
    _run(without, ModelFormat.GGUFQuantized)
    _run(at_zero, ModelFormat.GGUFQuantized, {"lora": (_lora(), 0.0)})
    _run(applied, ModelFormat.GGUFQuantized, {"lora": (_lora(), 0.5)})

    assert without.reserved[0].features["lora_sidecar_bytes"] == at_zero.reserved[0].features["lora_sidecar_bytes"] == 0
    # Its 24 elements, and per token of the joint sequence a row of proj_out's 4 outputs and of its 2 ranks, in the
    # compute dtype (float32 here).
    assert applied.reserved[0].features["lora_sidecar_bytes"] == (24 + _TOKENS * (4 + 2)) * 4


def test_a_dora_lora_on_quantized_layers_is_refused_before_the_transformer_is_loaded() -> None:
    """A DoRA layer normalizes the updated weight, which a Linear keeping its weight in buffers never forms. Here it is
    on BFL's output modulation, whose layer comes wrapped to swap its halves."""
    transformer = _Transformer()
    transformer.norm_out.linear = Int8ConvrotLinear(torch.ones(4, 8, dtype=torch.int8), torch.ones(4, 1), False)
    apply_custom_layers_to_model(transformer)
    native = "diffusion_model.final_layer.image.adaLN_modulation.1"
    low_rank = {f"{native}.lora_A.weight": torch.zeros(2, 8), f"{native}.lora_B.weight": torch.zeros(4, 2)}
    dora = lora_model_from_flux3_state_dict(
        low_rank | {f"{native}.lora_magnitude_vector.weight": torch.ones(4)}, name=_LORA_NAME
    )
    transformer_info = _TransformerInfo(transformer)

    with pytest.raises(ValueError, match=f"LoRA '{_LORA_NAME}' has DoRA layers.*quantized layers"):
        _run(transformer_info, ModelFormat.Checkpoint, {"lora": (dora, 1.0)})
    assert transformer_info.reserved == []
    # A low-rank LoRA there runs as a sidecar.
    plain = lora_model_from_flux3_state_dict(low_rank, name=_LORA_NAME)
    _run(transformer_info, ModelFormat.Checkpoint, {"lora": (plain, 1.0)})
    assert len(transformer_info.reserved) == 1
