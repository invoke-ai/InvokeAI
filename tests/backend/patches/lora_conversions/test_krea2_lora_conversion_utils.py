import accelerate
import pytest
import torch
from diffusers import Krea2Transformer2DModel

from invokeai.backend.model_manager.load.model_loaders.krea2 import KREA2_TRANSFORMER_CONFIG
from invokeai.backend.patches.layers.dora_layer import DoRALayer
from invokeai.backend.patches.layers.lokr_layer import LoKRLayer
from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.lora_conversions.krea2_lora_constants import (
    KREA2_LORA_QWEN3VL_PREFIX,
    KREA2_LORA_TRANSFORMER_PREFIX,
)
from invokeai.backend.patches.lora_conversions.krea2_lora_conversion_utils import (
    is_state_dict_likely_krea2_lora,
    lora_model_from_krea2_state_dict,
)
from tests.backend.patches.lora_conversions.lora_state_dicts.krea2_lora_kohya_format import (
    state_dict_keys as krea2_kohya_state_dict_keys,
)
from tests.backend.patches.lora_conversions.lora_state_dicts.utils import keys_to_mock_state_dict


def test_peft_layer_preserves_explicit_alpha() -> None:
    state_dict = {
        "transformer.text_fusion.0.attn.to_q.lora_A.weight": torch.ones(2, 4),
        "transformer.text_fusion.0.attn.to_q.lora_B.weight": torch.ones(4, 2),
        "transformer.text_fusion.0.attn.to_q.alpha": torch.tensor(1.0),
    }

    model = lora_model_from_krea2_state_dict(state_dict)

    layer = model.layers[f"{KREA2_LORA_TRANSFORMER_PREFIX}text_fusion.0.attn.to_q"]
    assert isinstance(layer, LoRALayer)
    assert layer._alpha == 1.0


def test_peft_dora_layer_preserves_magnitude_and_alpha() -> None:
    dora_scale = torch.full((4, 1), 2.0)
    state_dict = {
        "transformer.text_fusion.0.attn.to_q.lora_A.weight": torch.ones(2, 4),
        "transformer.text_fusion.0.attn.to_q.lora_B.weight": torch.ones(4, 2),
        "transformer.text_fusion.0.attn.to_q.dora_scale": dora_scale,
        "transformer.text_fusion.0.attn.to_q.alpha": torch.tensor(1.0),
    }

    model = lora_model_from_krea2_state_dict(state_dict)

    layer = model.layers[f"{KREA2_LORA_TRANSFORMER_PREFIX}text_fusion.0.attn.to_q"]
    assert isinstance(layer, DoRALayer)
    assert layer._alpha == 1.0
    assert torch.equal(layer.dora_scale, dora_scale)
    # `.dora_scale` is the LyCORIS magnitude: it indexes the *input* dim.
    assert layer.magnitude_is_out_dim is False


def test_peft_layer_without_explicit_alpha_uses_rank_default() -> None:
    state_dict = {
        "transformer.text_fusion.0.attn.to_q.lora_A.weight": torch.ones(2, 4),
        "transformer.text_fusion.0.attn.to_q.lora_B.weight": torch.ones(4, 2),
    }

    model = lora_model_from_krea2_state_dict(state_dict)

    layer = model.layers[f"{KREA2_LORA_TRANSFORMER_PREFIX}text_fusion.0.attn.to_q"]
    assert isinstance(layer, LoRALayer)
    assert layer._alpha is None


def test_incomplete_peft_pair_raises_descriptive_error() -> None:
    # A layer with lora_A but no matching lora_B is malformed. It must raise a clear ValueError naming the
    # missing key, not an uninformative bare KeyError.
    state_dict = {
        # Complete layer so the dict still looks like a Krea-2 LoRA.
        "transformer.text_fusion.0.attn.to_k.lora_A.weight": torch.ones(2, 4),
        "transformer.text_fusion.0.attn.to_k.lora_B.weight": torch.ones(4, 2),
        # Incomplete layer: lora_A present, lora_B missing.
        "transformer.text_fusion.0.attn.to_q.lora_A.weight": torch.ones(2, 4),
    }

    with pytest.raises(ValueError, match="lora_B.weight"):
        lora_model_from_krea2_state_dict(state_dict)


def test_peft_dora_magnitude_vector_key_produces_dora_layer() -> None:
    # Standard PEFT / Diffusers DoRA stores the magnitude as `.lora_magnitude_vector.weight` (not the LyCORIS
    # `.dora_scale`). It must be recognized and produce a DoRALayer preserving the magnitude, so valid
    # Diffusers DoRA adapters load instead of being split into a bogus, unrecognized layer (review 4802322488).
    magnitude = torch.full((4, 1), 3.0)
    state_dict = {
        "transformer.text_fusion.0.attn.to_q.lora_A.weight": torch.ones(2, 4),
        "transformer.text_fusion.0.attn.to_q.lora_B.weight": torch.ones(4, 2),
        "transformer.text_fusion.0.attn.to_q.lora_magnitude_vector.weight": magnitude,
    }

    model = lora_model_from_krea2_state_dict(state_dict)

    layer = model.layers[f"{KREA2_LORA_TRANSFORMER_PREFIX}text_fusion.0.attn.to_q"]
    assert isinstance(layer, DoRALayer)
    assert torch.equal(layer.dora_scale, magnitude)
    # The PEFT magnitude indexes the *output* dim, unlike the LyCORIS `.dora_scale`.
    assert layer.magnitude_is_out_dim is True


def test_native_aitoolkit_dora_magnitude_key_produces_dora_layer() -> None:
    # ai-toolkit (`network.type: dora`) writes native Krea-2 keys with a bare `.magnitude` suffix. Without an
    # explicit mapping these fall through the suffix table and get grouped into a bogus `...attn` layer,
    # raising "Unsupported lora format: dict_keys(['to_gate.magnitude', ...])" (issue #9515).
    attn_magnitude = torch.full((4,), 3.0)
    # A non-square layer: its magnitude has out_features entries while the LyCORIS convention would expect
    # in_features, so a mis-oriented magnitude would blow up at patch time rather than silently.
    ff_magnitude = torch.full((4,), 5.0)
    state_dict = {
        "diffusion_model.blocks.0.attn.wq.lora_A.weight": torch.ones(2, 4),
        "diffusion_model.blocks.0.attn.wq.lora_B.weight": torch.ones(4, 2),
        "diffusion_model.blocks.0.attn.wq.magnitude": attn_magnitude,
        "diffusion_model.txtfusion.refiner_blocks.0.mlp.down.lora_A.weight": torch.ones(2, 8),
        "diffusion_model.txtfusion.refiner_blocks.0.mlp.down.lora_B.weight": torch.ones(4, 2),
        "diffusion_model.txtfusion.refiner_blocks.0.mlp.down.magnitude": ff_magnitude,
    }

    model = lora_model_from_krea2_state_dict(state_dict)

    attn_layer = model.layers[f"{KREA2_LORA_TRANSFORMER_PREFIX}transformer_blocks.0.attn.to_q"]
    assert isinstance(attn_layer, DoRALayer)
    assert torch.equal(attn_layer.dora_scale, attn_magnitude)
    assert attn_layer.magnitude_is_out_dim is True

    ff_layer = model.layers[f"{KREA2_LORA_TRANSFORMER_PREFIX}text_fusion.refiner_blocks.0.ff.down"]
    assert isinstance(ff_layer, DoRALayer)
    assert torch.equal(ff_layer.dora_scale, ff_magnitude)
    assert ff_layer.magnitude_is_out_dim is True
    # The magnitude must survive as a real DoRA layer, not leak into a bogus parent group.
    assert not any(key.endswith(".attn") or key.endswith(".mlp") for key in model.layers)


def test_conflicting_transformer_and_diffusion_model_aliases_raise() -> None:
    # `transformer.` and `diffusion_model.` normalize to the same target key. Providing both aliases for one
    # logical layer (with different tensors) must raise, not silently drop one based on dict ordering.
    state_dict = {
        "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": torch.ones(2, 4),
        "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": torch.ones(4, 2),
        "diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight": torch.full((2, 4), 2.0),
        "diffusion_model.transformer_blocks.0.attn.to_q.lora_B.weight": torch.full((4, 2), 2.0),
    }

    with pytest.raises(ValueError, match="normalize to the same target"):
        lora_model_from_krea2_state_dict(state_dict)


def test_conflicting_native_and_diffusers_aliases_raise() -> None:
    state_dict = {
        "diffusion_model.blocks.0.attn.wq.lora_A.weight": torch.ones(2, 4),
        "diffusion_model.blocks.0.attn.wq.lora_B.weight": torch.ones(4, 2),
        "diffusion_model.blocks.0.attn.gate.lora_A.weight": torch.ones(2, 4),
        "diffusion_model.blocks.0.attn.gate.lora_B.weight": torch.ones(4, 2),
        "diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight": torch.full((2, 4), 2.0),
        "diffusion_model.transformer_blocks.0.attn.to_q.lora_B.weight": torch.full((4, 2), 2.0),
    }

    with pytest.raises(ValueError, match="normalize to the same target"):
        lora_model_from_krea2_state_dict(state_dict)


def test_native_comfyui_krea2_keys_are_remapped_to_diffusers_layout() -> None:
    # Native (ComfyUI) Krea-2 LoRAs name modules differently (blocks / attn.wq/wo/gate / mlp / txtfusion).
    # They must be remapped onto the diffusers Krea2Transformer2DModel layout so the LoRA actually applies.
    state_dict = {
        # main transformer block, native gated attention + SwiGLU mlp
        "diffusion_model.blocks.0.attn.wq.lora_A.weight": torch.ones(2, 4),
        "diffusion_model.blocks.0.attn.wq.lora_B.weight": torch.ones(4, 2),
        "diffusion_model.blocks.0.attn.wo.lora_A.weight": torch.ones(2, 4),
        "diffusion_model.blocks.0.attn.wo.lora_B.weight": torch.ones(4, 2),
        "diffusion_model.blocks.0.attn.gate.lora_A.weight": torch.ones(2, 4),
        "diffusion_model.blocks.0.attn.gate.lora_B.weight": torch.ones(4, 2),
        "diffusion_model.blocks.0.mlp.down.lora_A.weight": torch.ones(2, 8),
        "diffusion_model.blocks.0.mlp.down.lora_B.weight": torch.ones(4, 2),
        # text-fusion stage (native `txtfusion`, layerwise + refiner sub-blocks)
        "diffusion_model.txtfusion.layerwise_blocks.0.attn.wk.lora_A.weight": torch.ones(2, 4),
        "diffusion_model.txtfusion.layerwise_blocks.0.attn.wk.lora_B.weight": torch.ones(4, 2),
        "diffusion_model.txtfusion.refiner_blocks.1.mlp.up.lora_A.weight": torch.ones(2, 4),
        "diffusion_model.txtfusion.refiner_blocks.1.mlp.up.lora_B.weight": torch.ones(8, 2),
    }

    model = lora_model_from_krea2_state_dict(state_dict)
    keys = set(model.layers.keys())

    p = KREA2_LORA_TRANSFORMER_PREFIX
    # blocks -> transformer_blocks; attn.wq/wo/gate -> to_q/to_out.0/to_gate; mlp -> ff
    assert f"{p}transformer_blocks.0.attn.to_q" in keys
    assert f"{p}transformer_blocks.0.attn.to_out.0" in keys
    assert f"{p}transformer_blocks.0.attn.to_gate" in keys
    assert f"{p}transformer_blocks.0.ff.down" in keys
    # txtfusion -> text_fusion; layerwise_blocks / refiner_blocks preserved (NOT renamed to transformer_blocks)
    assert f"{p}text_fusion.layerwise_blocks.0.attn.to_k" in keys
    assert f"{p}text_fusion.refiner_blocks.1.ff.up" in keys
    # No native names should survive.
    assert not any(".wq" in k or ".wo" in k or ".mlp." in k or "txtfusion" in k for k in keys)
    assert all(isinstance(v, LoRALayer) for v in model.layers.values())


def test_native_krea2_dora_magnitude_is_preserved_through_remap() -> None:
    # A native DoRA slider (A/B + magnitude) must remap AND keep its magnitude, producing a DoRALayer.
    magnitude = torch.full((4, 1), 2.0)
    state_dict = {
        "diffusion_model.blocks.0.attn.wq.lora_A.weight": torch.ones(2, 4),
        "diffusion_model.blocks.0.attn.wq.lora_B.weight": torch.ones(4, 2),
        "diffusion_model.blocks.0.attn.wq.lora_magnitude_vector.weight": magnitude,
        # a second native marker so detection is unambiguous
        "diffusion_model.blocks.0.attn.gate.lora_A.weight": torch.ones(2, 4),
        "diffusion_model.blocks.0.attn.gate.lora_B.weight": torch.ones(4, 2),
    }

    model = lora_model_from_krea2_state_dict(state_dict)

    layer = model.layers[f"{KREA2_LORA_TRANSFORMER_PREFIX}transformer_blocks.0.attn.to_q"]
    assert isinstance(layer, DoRALayer)
    assert torch.equal(layer.dora_scale, magnitude)


def test_diffusers_layout_krea2_keys_are_left_untouched() -> None:
    # A LoRA already in the diffusers layout must not be altered by the native remap.
    state_dict = {
        "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": torch.ones(2, 4),
        "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": torch.ones(4, 2),
        "transformer.text_fusion.0.attn.to_q.lora_A.weight": torch.ones(2, 4),
        "transformer.text_fusion.0.attn.to_q.lora_B.weight": torch.ones(4, 2),
    }

    model = lora_model_from_krea2_state_dict(state_dict)
    keys = set(model.layers.keys())
    p = KREA2_LORA_TRANSFORMER_PREFIX
    assert f"{p}transformer_blocks.0.attn.to_q" in keys
    assert f"{p}text_fusion.0.attn.to_q" in keys


@pytest.mark.parametrize(
    ("native_module", "diffusers_module"),
    [
        ("blocks.0.attn.wq", "transformer_blocks.0.attn.to_q"),
        ("blocks.0.mlp.down", "transformer_blocks.0.ff.down"),
    ],
)
def test_single_module_native_krea2_lora_is_remapped(
    native_module: str,
    diffusers_module: str,
) -> None:
    state_dict = {
        f"diffusion_model.{native_module}.lora_A.weight": torch.ones(2, 4),
        f"diffusion_model.{native_module}.lora_B.weight": torch.ones(4, 2),
    }

    model = lora_model_from_krea2_state_dict(state_dict)

    assert set(model.layers) == {f"{KREA2_LORA_TRANSFORMER_PREFIX}{diffusers_module}"}


@pytest.mark.parametrize(
    ("kohya_module", "diffusers_module"),
    [
        ("blocks_0_attn_wq", "transformer_blocks.0.attn.to_q"),
        ("blocks_0_attn_wk", "transformer_blocks.0.attn.to_k"),
        ("blocks_0_attn_wv", "transformer_blocks.0.attn.to_v"),
        ("blocks_0_attn_wo", "transformer_blocks.0.attn.to_out.0"),
        ("blocks_0_attn_gate", "transformer_blocks.0.attn.to_gate"),
        # Multi-digit block index.
        ("blocks_27_mlp_down", "transformer_blocks.27.ff.down"),
        # `layerwise_blocks` / `refiner_blocks` are the native components that themselves contain an
        # underscore, i.e. the only genuine ambiguity in the flattened form.
        ("txtfusion_layerwise_blocks_0_attn_wo", "text_fusion.layerwise_blocks.0.attn.to_out.0"),
        ("txtfusion_refiner_blocks_1_mlp_gate", "text_fusion.refiner_blocks.1.ff.gate"),
        ("txtfusion_projector", "text_fusion.projector"),
        ("first", "img_in"),
        ("tmlp_0", "time_embed.linear_1"),
        ("tmlp_2", "time_embed.linear_2"),
        ("tproj_1", "time_mod_proj"),
        ("txtmlp_1", "txt_in.linear_1"),
        ("txtmlp_3", "txt_in.linear_2"),
        ("last_linear", "final_layer.linear"),
    ],
)
def test_kohya_flattened_krea2_module_is_remapped(kohya_module: str, diffusers_module: str) -> None:
    # kohya / LyCORIS flatten the module path and prefix it with `lora_unet_`. Without un-flattening, every key
    # misses its module and the adapter is a silent no-op ("Failed to find module for LoRA layer key:
    # lora_transformer-lora_unet_blocks_6_attn_wv").
    state_dict = {
        f"lora_unet_{kohya_module}.lora_down.weight": torch.ones(2, 4),
        f"lora_unet_{kohya_module}.lora_up.weight": torch.ones(4, 2),
    }

    model = lora_model_from_krea2_state_dict(state_dict)

    assert set(model.layers) == {f"{KREA2_LORA_TRANSFORMER_PREFIX}{diffusers_module}"}


def test_kohya_krea2_lora_layers_match_the_real_transformer() -> None:
    # Every layer of a real kohya Krea-2 adapter must land on an actual Linear of Krea2Transformer2DModel, with
    # in/out features that agree with the LoRA's own down/up shapes. A wrong rename (e.g. ff.gate <-> ff.down,
    # whose SwiGLU shapes are transposed) is caught here rather than as a runtime warning.
    state_dict = keys_to_mock_state_dict(krea2_kohya_state_dict_keys)

    model = lora_model_from_krea2_state_dict(state_dict)

    with accelerate.init_empty_weights():
        transformer = Krea2Transformer2DModel(**KREA2_TRANSFORMER_CONFIG)

    # 4 blocks x 8 Linears (2 transformer, 1 layerwise + 1 refiner text-fusion) + 8 top-level modules.
    assert len(model.layers) == 40
    for layer_key, layer in model.layers.items():
        module_name = layer_key[len(KREA2_LORA_TRANSFORMER_PREFIX) :]
        submodule = transformer.get_submodule(module_name)
        assert isinstance(submodule, torch.nn.Linear), f"{module_name} is not a Linear"
        out_features, in_features = submodule.weight.shape
        assert layer.down.shape[1] == in_features, f"{module_name} in_features mismatch"
        assert layer.up.shape[0] == out_features, f"{module_name} out_features mismatch"


def test_kohya_flattened_krea2_layer_preserves_alpha() -> None:
    # kohya adapters carry an explicit `.alpha`; it must survive the un-flattening intact, otherwise the LoRA
    # applies at the wrong strength.
    state_dict = {
        "lora_unet_blocks_6_attn_wv.lora_down.weight": torch.ones(2, 4),
        "lora_unet_blocks_6_attn_wv.lora_up.weight": torch.ones(4, 2),
        "lora_unet_blocks_6_attn_wv.alpha": torch.tensor(2.0),
    }

    model = lora_model_from_krea2_state_dict(state_dict)

    layer = model.layers[f"{KREA2_LORA_TRANSFORMER_PREFIX}transformer_blocks.6.attn.to_v"]
    assert isinstance(layer, LoRALayer)
    assert layer._alpha == 2.0


def test_kohya_flattened_krea2_keys_tolerate_doubled_separator() -> None:
    state_dict = {
        "lora_unet__blocks_6_attn_wv.lora_down.weight": torch.ones(2, 4),
        "lora_unet__blocks_6_attn_wv.lora_up.weight": torch.ones(4, 2),
    }

    model = lora_model_from_krea2_state_dict(state_dict)

    assert set(model.layers) == {f"{KREA2_LORA_TRANSFORMER_PREFIX}transformer_blocks.6.attn.to_v"}


@pytest.mark.parametrize(
    "flat_module",
    [
        # Non-Linear natives: `mod.lin` is folded into the `scale_shift_table` parameter and the norms have no
        # Linear counterpart, so there is nothing to patch. They must not be renamed into a key that merely
        # looks applicable.
        "blocks_6_mod_lin",
        "blocks_6_prenorm",
        "blocks_6_attn_qknorm_qnorm",
        # Not a Krea-2 module layout at all (e.g. a flattened adapter for some other architecture).
        "double_blocks_0_img_attn_proj",
        # Sequential positions that hold an activation rather than a Linear. The parsing tree enumerates the
        # indices it accepts, so these are rejected outright instead of being rewritten to `tmlp.1.*` — a
        # half-converted key that the native pass no longer recognizes.
        "tmlp_1",
        "tproj_0",
        "txtmlp_2",
    ],
)
def test_unrecognized_kohya_flattened_keys_are_left_untouched(flat_module: str) -> None:
    state_dict = {
        f"lora_unet_{flat_module}.lora_down.weight": torch.ones(2, 4),
        f"lora_unet_{flat_module}.lora_up.weight": torch.ones(4, 2),
    }

    model = lora_model_from_krea2_state_dict(state_dict)

    # Left verbatim rather than rewritten into a plausible-looking but wrong module path.
    assert set(model.layers) == {f"{KREA2_LORA_TRANSFORMER_PREFIX}lora_unet_{flat_module}"}


@pytest.mark.parametrize(
    ("lycoris_suffixes", "expected_second_layer"),
    [
        # LoKr is a layout this converter understands, so its module un-flattens like any other and the
        # adapter actually applies. The other algorithms below have no handler, so they stay verbatim.
        (("lokr_w1", "lokr_w2"), "transformer_blocks.6.attn.to_q"),
        (("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b"), "lora_unet_blocks_6_attn_wq"),
        (("diff", "diff_b"), "lora_unet_blocks_6_attn_wq"),
        # LyCORIS saves an `alpha` per module, so this is the realistic on-disk shape rather than an
        # edge case — see this repo's own captured fixtures. `.alpha` is a suffix the converter knows,
        # so deciding per key rewrote it while its siblings stayed verbatim, splitting one module into
        # two groups and aborting the load on the orphaned `{'alpha'}`.
        (("lokr_w1", "lokr_w2", "alpha"), "transformer_blocks.6.attn.to_q"),
        (("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b", "alpha"), "lora_unet_blocks_6_attn_wq"),
        # `dora_scale` is deliberately not combined with a LyCORIS algorithm here: it would orphan the
        # same way, but even grouped correctly `any_lora_layer_from_state_dict` tests `dora_scale`
        # before `lokr_w1`, so a weight-decomposed LoKr dispatches to DoRALayer and dies on a missing
        # `lora_up.weight`. That precedence is shared code, predates this branch, and is not what the
        # per-module gate below is about.
    ],
)
def test_kohya_lycoris_algorithm_keys_do_not_abort_the_load(
    lycoris_suffixes: tuple[str, ...], expected_second_layer: str
) -> None:
    # LyCORIS supports per-module algorithms, so one kohya file can mix ordinary lora_down/up modules with
    # LoKr/LoHa/full ones. Un-flattening a key whose suffix `_group_by_layer` cannot split back off used to
    # feed it a dotted path, whose blind `rsplit(".", 2)` fallback then cut inside the module name and fused
    # two modules into one unsupported layer — aborting the *entire* adapter at generation time.
    # `alpha` is a scalar on disk and `dora_scale` a per-channel vector; giving them weight-shaped
    # tensors would fail inside the layer for reasons that have nothing to do with the grouping.
    shapes = {"alpha": torch.tensor(4.0), "dora_scale": torch.ones(4)}
    state_dict = {
        "lora_unet_blocks_0_attn_wv.lora_down.weight": torch.ones(2, 4),
        "lora_unet_blocks_0_attn_wv.lora_up.weight": torch.ones(4, 2),
        **{f"lora_unet_blocks_6_attn_wq.{suffix}": shapes.get(suffix, torch.ones(4, 4)) for suffix in lycoris_suffixes},
    }

    model = lora_model_from_krea2_state_dict(state_dict)

    # Either way the load completes: a supported algorithm converts onto its real module, an unsupported one
    # stays verbatim and degrades to the per-layer "Failed to find module" warning at apply time, rather than
    # taking the whole adapter down.
    assert set(model.layers) == {
        f"{KREA2_LORA_TRANSFORMER_PREFIX}transformer_blocks.0.attn.to_v",
        f"{KREA2_LORA_TRANSFORMER_PREFIX}{expected_second_layer}",
    }


def test_non_string_keys_survive_the_kohya_and_native_passes() -> None:
    # `.pt`/`.ckpt` sources can carry non-string keys. Once the kohya pass rewrites something, the native pass
    # runs its substring tests over every key — which raised `TypeError: argument of type 'int' is not
    # iterable` on an int key rather than leaving it alone.
    state_dict = {
        0: torch.ones(2),
        "lora_unet_blocks_0_attn_wv.lora_down.weight": torch.ones(2, 4),
        "lora_unet_blocks_0_attn_wv.lora_up.weight": torch.ones(4, 2),
    }

    model = lora_model_from_krea2_state_dict(state_dict)

    assert f"{KREA2_LORA_TRANSFORMER_PREFIX}transformer_blocks.0.attn.to_v" in model.layers


def test_conflicting_kohya_and_native_aliases_raise() -> None:
    # The flattened and dotted spellings of one logical layer normalize to the same target. Providing both
    # must raise instead of silently dropping one based on dict ordering.
    state_dict = {
        "lora_unet_blocks_0_attn_wq.lora_down.weight": torch.ones(2, 4),
        "lora_unet_blocks_0_attn_wq.lora_up.weight": torch.ones(4, 2),
        "blocks.0.attn.wq.lora_down.weight": torch.full((2, 4), 2.0),
        "blocks.0.attn.wq.lora_up.weight": torch.full((4, 2), 2.0),
    }

    with pytest.raises(ValueError, match="normalize to the same target"):
        lora_model_from_krea2_state_dict(state_dict)


def test_native_transformer_remap_does_not_change_diffusers_text_encoder_blocks() -> None:
    state_dict = {
        "diffusion_model.blocks.0.attn.wq.lora_A.weight": torch.ones(2, 4),
        "diffusion_model.blocks.0.attn.wq.lora_B.weight": torch.ones(4, 2),
        "text_encoder.visual.blocks.0.attn.qkv.lora_A.weight": torch.ones(2, 4),
        "text_encoder.visual.blocks.0.attn.qkv.lora_B.weight": torch.ones(4, 2),
    }

    model = lora_model_from_krea2_state_dict(state_dict)

    assert f"{KREA2_LORA_TRANSFORMER_PREFIX}transformer_blocks.0.attn.to_q" in model.layers
    assert f"{KREA2_LORA_QWEN3VL_PREFIX}visual.blocks.0.attn.qkv" in model.layers


def test_native_krea2_top_level_linear_keys_are_remapped() -> None:
    native_to_diffusers = {
        "first": "img_in",
        "tmlp.0": "time_embed.linear_1",
        "tmlp.2": "time_embed.linear_2",
        "tproj.1": "time_mod_proj",
        "txtmlp.1": "txt_in.linear_1",
        "txtmlp.3": "txt_in.linear_2",
        "last.linear": "final_layer.linear",
    }
    state_dict = {
        f"diffusion_model.{module}.{suffix}.weight": torch.ones(2, 4)
        for module in native_to_diffusers
        for suffix in ("lora_A", "lora_B")
    }
    state_dict.update(
        {
            "diffusion_model.txtfusion.layerwise_blocks.0.attn.wq.lora_A.weight": torch.ones(2, 4),
            "diffusion_model.txtfusion.layerwise_blocks.0.attn.wq.lora_B.weight": torch.ones(4, 2),
        }
    )

    model = lora_model_from_krea2_state_dict(state_dict)

    expected_keys = {
        f"{KREA2_LORA_TRANSFORMER_PREFIX}{diffusers_module}" for diffusers_module in native_to_diffusers.values()
    }
    assert expected_keys < set(model.layers)


def test_lokr_layer_produces_lokr_layer() -> None:
    # LyCORIS LoKr adapters (e.g. those produced by ai-toolkit for Krea-2) carry Kronecker factors instead of
    # a lora_A/lora_B pair. They must survive _group_by_layer intact so any_lora_layer_from_state_dict can
    # route them to LoKRLayer.
    state_dict = {
        "transformer.text_fusion.0.attn.to_q.lokr_w1": torch.ones(2, 2),
        "transformer.text_fusion.0.attn.to_q.lokr_w2": torch.ones(3, 4),
        "transformer.text_fusion.0.attn.to_q.alpha": torch.tensor(1.0),
    }

    model = lora_model_from_krea2_state_dict(state_dict)

    layer = model.layers[f"{KREA2_LORA_TRANSFORMER_PREFIX}text_fusion.0.attn.to_q"]
    assert isinstance(layer, LoKRLayer)
    assert layer._alpha == 1.0
    # The reconstructed weight is the Kronecker product of the two factors.
    assert layer.get_weight(torch.empty(6, 8)).shape == (6, 8)


def test_factored_lokr_layer_produces_lokr_layer() -> None:
    # LoKr may factor either Kronecker operand further into an `_a`/`_b` pair. Both spellings must be grouped
    # onto the same layer.
    state_dict = {
        "transformer.text_fusion.0.attn.to_q.lokr_w1_a": torch.ones(2, 1),
        "transformer.text_fusion.0.attn.to_q.lokr_w1_b": torch.ones(1, 2),
        "transformer.text_fusion.0.attn.to_q.lokr_w2_a": torch.ones(3, 1),
        "transformer.text_fusion.0.attn.to_q.lokr_w2_b": torch.ones(1, 4),
    }

    model = lora_model_from_krea2_state_dict(state_dict)

    layer = model.layers[f"{KREA2_LORA_TRANSFORMER_PREFIX}text_fusion.0.attn.to_q"]
    assert isinstance(layer, LoKRLayer)
    assert layer.w1_a is not None and layer.w1_b is not None
    assert layer.w2_a is not None and layer.w2_b is not None


def test_native_lokr_keys_are_renamed_to_diffusers_layout() -> None:
    # Native (ComfyUI / ai-toolkit) LoKr keys must go through the same native->diffusers renaming as LoRA
    # keys: txtfusion -> text_fusion, attn.wq -> attn.to_q, mlp.down -> ff.down.
    state_dict = {
        "diffusion_model.txtfusion.layerwise_blocks.0.attn.wq.lokr_w1": torch.ones(2, 2),
        "diffusion_model.txtfusion.layerwise_blocks.0.attn.wq.lokr_w2": torch.ones(3, 4),
        "diffusion_model.txtfusion.refiner_blocks.1.mlp.down.lokr_w1": torch.ones(2, 2),
        "diffusion_model.txtfusion.refiner_blocks.1.mlp.down.lokr_w2": torch.ones(3, 4),
    }

    model = lora_model_from_krea2_state_dict(state_dict)

    assert set(model.layers) == {
        f"{KREA2_LORA_TRANSFORMER_PREFIX}text_fusion.layerwise_blocks.0.attn.to_q",
        f"{KREA2_LORA_TRANSFORMER_PREFIX}text_fusion.refiner_blocks.1.ff.down",
    }
    assert all(isinstance(layer, LoKRLayer) for layer in model.layers.values())


def test_is_state_dict_likely_krea2_lora_accepts_lokr() -> None:
    state_dict = {
        "diffusion_model.txtfusion.layerwise_blocks.0.attn.wq.lokr_w1": torch.ones(2, 2),
        "diffusion_model.txtfusion.layerwise_blocks.0.attn.wq.lokr_w2": torch.ones(3, 4),
    }

    assert is_state_dict_likely_krea2_lora(state_dict)


def test_is_state_dict_likely_krea2_lora_rejects_lokr_without_krea2_modules() -> None:
    # The Krea-2 signature modules are still required: a LoKr targeting only generic transformer blocks
    # belongs to another base (e.g. Qwen-Image) and must not be claimed here.
    state_dict = {
        "transformer.transformer_blocks.0.attn.to_q.lokr_w1": torch.ones(2, 2),
        "transformer.transformer_blocks.0.attn.to_q.lokr_w2": torch.ones(3, 4),
    }

    assert not is_state_dict_likely_krea2_lora(state_dict)


def test_kohya_flattened_lokr_converts_onto_its_real_module() -> None:
    # A LoKr adapter saved in the kohya flattened layout has to clear both hurdles at once: the un-flattening
    # pass has to reconstruct the dotted module path, and the grouper has to recognise the `lokr_*` suffixes.
    # Before LoKr was a known suffix the module was left verbatim and the adapter was a silent no-op.
    state_dict = {
        "lora_unet_txtfusion_layerwise_blocks_0_attn_wq.lokr_w1": torch.ones(2, 2),
        "lora_unet_txtfusion_layerwise_blocks_0_attn_wq.lokr_w2": torch.ones(3, 4),
        "lora_unet_txtfusion_layerwise_blocks_0_attn_wq.alpha": torch.tensor(4.0),
    }

    model = lora_model_from_krea2_state_dict(state_dict)

    layer = model.layers[f"{KREA2_LORA_TRANSFORMER_PREFIX}text_fusion.layerwise_blocks.0.attn.to_q"]
    assert isinstance(layer, LoKRLayer)
    assert layer.get_weight(torch.empty(6, 8)).shape == (6, 8)
