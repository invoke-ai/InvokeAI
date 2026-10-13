"""How `Ideogram4DenoiseInvocation` applies the LoRAs on 'Transformer' to its two branches.

Every published Ideogram 4 LoRA is one set of weights, and the node applies it to both branches: rendered side by
side, patching only the conditional one let guidance scale the LoRA's whole effect. Both builds take the same path --
two single files (two cache records) and a Diffusers pair (one record holding both) -- and each branch has to come
back unpatched when the loop ends.
"""

from contextlib import ExitStack, contextmanager
from types import SimpleNamespace

import pytest
import torch

from invokeai.app.invocations.ideogram4.ideogram4_denoise import Ideogram4DenoiseInvocation
from invokeai.app.invocations.model import LoRAField, ModelIdentifierField, TransformerField
from invokeai.backend.ideogram4.modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer
from invokeai.backend.ideogram4.transformer_pair import Ideogram4TransformerPair
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType
from invokeai.backend.patches.layers.lokr_layer import LoKRLayer
from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.lora_conversions.ideogram4_lora_constants import IDEOGRAM4_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw

TINY = Ideogram4Config(
    emb_dim=64,
    num_layers=1,
    num_heads=2,
    intermediate_size=128,
    adanln_dim=16,
    in_channels=8,
    llm_features_dim=32,
    mrope_section=(4, 2, 2),
)
QKV = "layers.0.attention.qkv"


def _identifier(key: str, model_type: ModelType) -> ModelIdentifierField:
    return ModelIdentifierField(key=key, hash=f"hash-{key}", name=key, base=BaseModelType.Ideogram4, type=model_type)


def _field(key: str, loras: list[LoRAField] | None = None) -> TransformerField:
    return TransformerField(transformer=_identifier(key, ModelType.Main), loras=loras or [])


def _lora(key: str, weight: float = 1.0) -> LoRAField:
    return LoRAField(lora=_identifier(key, ModelType.LoRA), weight=weight)


class _Loaded:
    """The slice of `LoadedModel` the node uses: an unlocked `model`, a lock, and a RAM pin for LoRAs."""

    def __init__(self, model, calls: list[tuple[str, int]], key: str, cached_weights=None) -> None:
        self.model = model
        self._calls = calls
        self._key = key
        self._cached_weights = cached_weights
        self.compute_device = torch.device("cpu")
        self.supports_partial_loading = False

    @contextmanager
    def model_on_device(self, working_mem_bytes: int = 0):
        self._calls.append((self._key, working_mem_bytes))
        yield (self._cached_weights, self.model)

    @contextmanager
    def model_in_ram(self):
        yield self.model


def _context(models: dict, formats: dict[str, ModelFormat], calls: list | None = None, cached: dict | None = None):
    recorded = calls if calls is not None else []
    return SimpleNamespace(
        models=SimpleNamespace(
            load=lambda identifier: _Loaded(
                models[identifier.key], recorded, identifier.key, (cached or {}).get(identifier.key)
            ),
            get_config=lambda identifier: SimpleNamespace(format=formats[identifier.key]),
        ),
        logger=SimpleNamespace(info=lambda *_: None, warning=lambda *_: None),
        config=SimpleNamespace(get=lambda: SimpleNamespace(device_working_mem_gb=3.0, max_cache_vram_gb=None)),
    )


def _transformer(seed: int) -> Ideogram4Transformer:
    torch.manual_seed(seed)
    model = Ideogram4Transformer(TINY)
    apply_custom_layers_to_model(model)
    return model


def _qkv_lora(seed: int) -> tuple[ModelPatchRaw, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    up = torch.randn(3 * TINY.emb_dim, 4, generator=generator)
    down = torch.randn(4, TINY.emb_dim, generator=generator)
    layer = LoRALayer(up=up, mid=None, down=down, alpha=None, bias=None)
    return ModelPatchRaw(layers={f"{IDEOGRAM4_LORA_TRANSFORMER_PREFIX}{QKV}": layer}), up @ down


def _qkv(model: Ideogram4Transformer, x: torch.Tensor) -> torch.Tensor:
    return model.get_submodule(QKV)(x)


def test_two_single_files_both_take_the_lora_and_give_it_back() -> None:
    """On the CPU the patcher applies every layer as a sidecar, so direct patching only shows on a GPU; what this
    pins is that both branches get the patch, scaled by the LoRA's weight, and lose it again."""
    model_format = ModelFormat.GGUFQuantized
    patch, delta = _qkv_lora(seed=1)
    models = {"cond": _transformer(0), "uncond": _transformer(1), "lora": patch}
    weights = {branch: models[branch].get_submodule(QKV).weight.detach().clone() for branch in ("cond", "uncond")}
    invocation = Ideogram4DenoiseInvocation.model_construct(
        transformer=_field("cond", [_lora("lora", weight=0.5)]), unconditional_transformer=_field("uncond")
    )
    context = _context(models, {"cond": model_format, "uncond": model_format, "lora": ModelFormat.LyCORIS})
    x = torch.randn(2, TINY.emb_dim)

    with ExitStack() as stack:
        conditional, unconditional = invocation._load_branches(context, stack, 0)
        for branch, model in (("cond", conditional), ("uncond", unconditional)):
            torch.testing.assert_close(_qkv(model, x), x @ (weights[branch] + 0.5 * delta).T)

    for branch in ("cond", "uncond"):
        torch.testing.assert_close(_qkv(models[branch], x), x @ weights[branch].T)


def test_a_pair_patches_both_of_its_branches_and_gives_them_back() -> None:
    pair = Ideogram4TransformerPair(conditional=_transformer(0), unconditional=_transformer(1))
    patch, delta = _qkv_lora(seed=1)
    invocation = Ideogram4DenoiseInvocation.model_construct(
        transformer=_field("pair", [_lora("lora")]), unconditional_transformer=None
    )
    context = _context({"pair": pair, "lora": patch}, {"pair": ModelFormat.Diffusers, "lora": ModelFormat.LyCORIS})
    x = torch.randn(2, TINY.emb_dim)
    live = {
        branch: getattr(pair, branch).get_submodule(QKV).weight.detach().clone()
        for branch in ("conditional", "unconditional")
    }

    with ExitStack() as stack:
        invocation._load_branches(context, stack, 0)
        for branch in ("conditional", "unconditional"):
            torch.testing.assert_close(_qkv(getattr(pair, branch), x), x @ (live[branch] + delta).T)

    for branch in ("conditional", "unconditional"):
        torch.testing.assert_close(_qkv(getattr(pair, branch), x), x @ live[branch].T)


def test_a_directly_patched_pair_restores_each_branch_from_its_own_half_of_the_ram_copy(monkeypatch) -> None:
    """One cache record holds both branches, so its RAM copy is keyed `conditional.` / `unconditional.`. A directly
    patched branch is restored from the copy it is handed, so each has to get its own half, keyed like its own
    weights. The copy here is deliberately distinct per branch and from the live weights, so a restore from anything
    else shows. The patcher only patches in place on a GPU; its CPU fallback to the sidecar is switched off."""
    from invokeai.backend.patches.layer_patcher import LayerPatcher

    monkeypatch.setattr(LayerPatcher, "_is_any_part_of_layer_on_cpu", staticmethod(lambda module: False))
    pair = Ideogram4TransformerPair(conditional=_transformer(0), unconditional=_transformer(1))
    patch, delta = _qkv_lora(seed=1)
    ram_copy = {key: value.detach().clone() for key, value in pair.state_dict().items()}
    restored_to = {"conditional": 0.25, "unconditional": 0.5}
    for branch, value in restored_to.items():
        ram_copy[f"{branch}.{QKV}.weight"] = torch.full_like(ram_copy[f"{branch}.{QKV}.weight"], value)
    live = {branch: getattr(pair, branch).get_submodule(QKV).weight.detach().clone() for branch in restored_to}
    invocation = Ideogram4DenoiseInvocation.model_construct(
        transformer=_field("pair", [_lora("lora")]), unconditional_transformer=None
    )
    context = _context(
        {"pair": pair, "lora": patch},
        {"pair": ModelFormat.Diffusers, "lora": ModelFormat.LyCORIS},
        cached={"pair": ram_copy},
    )

    with ExitStack() as stack:
        invocation._load_branches(context, stack, 0)
        for branch in restored_to:
            torch.testing.assert_close(getattr(pair, branch).get_submodule(QKV).weight, live[branch] + delta)

    for branch, value in restored_to.items():
        weight = getattr(pair, branch).get_submodule(QKV).weight
        torch.testing.assert_close(weight, torch.full_like(weight, value))


def test_loras_on_the_unconditional_input_are_refused() -> None:
    """They would otherwise be ignored without a word; the node applies the 'Transformer' LoRAs to both branches."""
    invocation = Ideogram4DenoiseInvocation.model_construct(
        transformer=_field("cond"), unconditional_transformer=_field("uncond", [_lora("lora")])
    )
    context = _context({"cond": _transformer(0), "uncond": _transformer(1)}, {})

    with pytest.raises(ValueError, match="Transformer \\(Unconditional\\)"), ExitStack() as stack:
        invocation._load_branches(context, stack, 0)


def _lokr_on_qkv() -> ModelPatchRaw:
    layer = LoKRLayer(
        w1=torch.ones(3, 1),
        w1_a=None,
        w1_b=None,
        w2=torch.ones(TINY.emb_dim, TINY.emb_dim) * 0.01,
        w2_a=None,
        w2_b=None,
        t2=None,
        alpha=None,
        bias=None,
    )
    return ModelPatchRaw(layers={f"{IDEOGRAM4_LORA_TRANSFORMER_PREFIX}{QKV}": layer})


def _reservation(
    model_format: ModelFormat, lora: ModelPatchRaw | None, *, pair: bool = False, fp8: bool = False
) -> int:
    """The working memory the node locks its first record with, for two single files or one pair.

    `fp8` stores every branch's `qkv` weight as float8, as FP8 Storage leaves a checkpoint's Linears.
    """
    branches = {"cond": _transformer(0), "uncond": _transformer(1)}
    if fp8:
        for model in branches.values():
            qkv = model.get_submodule(QKV)
            qkv.weight = torch.nn.Parameter(qkv.weight.detach().to(torch.float8_e4m3fn), requires_grad=False)
    loras = [_lora("lora")] if lora is not None else []
    calls: list[tuple[str, int]] = []
    if pair:
        models = {"pair": Ideogram4TransformerPair(branches["cond"], branches["uncond"]), "lora": lora}
        invocation = Ideogram4DenoiseInvocation.model_construct(
            transformer=_field("pair", loras), unconditional_transformer=None
        )
        context = _context(models, {"pair": model_format, "lora": ModelFormat.LyCORIS}, calls)
    else:
        models = {**branches, "lora": lora}
        invocation = Ideogram4DenoiseInvocation.model_construct(
            transformer=_field("cond", loras), unconditional_transformer=_field("uncond")
        )
        context = _context(models, {"cond": model_format, "uncond": model_format, "lora": ModelFormat.LyCORIS}, calls)
    with ExitStack() as stack:
        invocation._load_branches(context, stack, 4 * 1024**3)
    return calls[0][1]


@pytest.mark.parametrize("pair", [False, True], ids=["single-files", "pair"])
def test_sidecar_loras_are_reserved_once_for_both_branches(pair: bool) -> None:
    """The patch object both branches share stays on the device for the loop, counted once. A low-rank layer adds
    nothing beyond it: its residual is an activation-sized copy, inside the per-token estimate."""
    patch, _delta = _qkv_lora(seed=1)

    extra = _reservation(ModelFormat.GGUFQuantized, patch, pair=pair) - _reservation(
        ModelFormat.GGUFQuantized, None, pair=pair
    )

    assert extra == patch.calc_size()


@pytest.mark.parametrize("pair", [False, True], ids=["single-files", "pair"])
def test_a_lokr_layer_also_reserves_the_weight_it_materializes(pair: bool) -> None:
    """A LoKR layer is applied in weight space: the module's weight, its delta and their sum, up to float32."""
    lokr = _lokr_on_qkv()

    extra = _reservation(ModelFormat.GGUFQuantized, lokr, pair=pair) - _reservation(
        ModelFormat.GGUFQuantized, None, pair=pair
    )

    assert extra == lokr.calc_size() + 3 * 4 * (3 * TINY.emb_dim) * TINY.emb_dim


def test_fp8_weights_reserve_their_loras_too() -> None:
    """Nothing about the format says sidecar, but fp8 weights have no add kernel, so the patcher picks it itself."""
    patch, _delta = _qkv_lora(seed=1)

    extra = _reservation(ModelFormat.Checkpoint, patch, fp8=True) - _reservation(ModelFormat.Checkpoint, None, fp8=True)

    assert extra == patch.calc_size()


def test_a_directly_patched_pair_of_files_reserves_nothing_for_its_loras() -> None:
    patch, _delta = _qkv_lora(seed=1)

    assert _reservation(ModelFormat.Checkpoint, patch) == _reservation(ModelFormat.Checkpoint, None)


def test_patches_take_the_dtype_the_branch_computes_in() -> None:
    """A quantized branch's first parameter is a packed weight (uint8 for nf4). Casting the LoRA to that would wreck
    it -- and the cache's own copy of it, which the patcher casts back afterwards."""
    patch, delta = _qkv_lora(seed=1)
    original_down = patch.layers[f"{IDEOGRAM4_LORA_TRANSFORMER_PREFIX}{QKV}"].down.clone()
    models = {"cond": _transformer(0), "uncond": _transformer(1), "lora": patch}
    for model in (models["cond"], models["uncond"]):
        model.input_proj.weight = torch.nn.Parameter(
            torch.zeros_like(model.input_proj.weight, dtype=torch.uint8), requires_grad=False
        )
        model.input_proj.compute_dtype = torch.float32
    weight = models["cond"].get_submodule(QKV).weight.detach().clone()
    invocation = Ideogram4DenoiseInvocation.model_construct(
        transformer=_field("cond", [_lora("lora")]), unconditional_transformer=_field("uncond")
    )
    context = _context(
        models, {"cond": ModelFormat.Checkpoint, "uncond": ModelFormat.Checkpoint, "lora": ModelFormat.LyCORIS}
    )
    x = torch.randn(2, TINY.emb_dim)

    with ExitStack() as stack:
        conditional, _unconditional = invocation._load_branches(context, stack, 0)
        torch.testing.assert_close(_qkv(conditional, x), x @ (weight + delta).T)

    torch.testing.assert_close(patch.layers[f"{IDEOGRAM4_LORA_TRANSFORMER_PREFIX}{QKV}"].down, original_down)
