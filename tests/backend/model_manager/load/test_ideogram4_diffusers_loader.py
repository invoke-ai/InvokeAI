"""Loader-level tests for the diffusers-folder Ideogram 4 path.

Ideogram 4 has two loaders. The single-file one (`Ideogram4CheckpointModel._load_model`)
rules int8 out before it probes for scales, and says why in a comment there: an int8 layer ships a
`.weight_scale` too, so probing for scales without ruling int8 out "would take the whole checkpoint
down the fp8 path -- scaled, but never un-rotated, which loads cleanly and generates noise."

The folder loader had no such branch, and three of its seams took a foreign payload quietly.

**The transformer.** Its probe is `is_fp8_state_dict`, satisfied by any key ending `.weight_scale`
whatever the weight's dtype, and what it selects is Ideogram's own weight-only fp8: an `Fp8Linear`
whose forward is `weight.to(dtype) * weight_scale.unsqueeze(1)` -- a fold, deferred to forward.
Driven with a rotated `int8_tensorwise` payload, the layer it produced correlated with the weight
the file encodes at **0.067** on this file's geometry. The same seam also dropped the *other*
spelling of the scale: `swap_linears_to_fp8` gates on `.weight_scale` alone, so a `.scale_weight`
file installs no `Fp8Linear` at all and the raw fp8 codes are copied into an ordinary `nn.Linear`.

**The text encoder.** Its fp8 branch is gated on `ideogram_fp8_weight_only`, a flag InvokeAI writes
into `config.json` and no one else does, so any third-party fp8 repack falls through to
`load_state_dict(strict=False, assign=True)` -- where the codes *become* the parameters, off by
`1/weight_scale`, measured at 6803x for the payload below, with the scale reported at DEBUG. That
branch is one of nine call sites of `reject_quantized_side_channel`; the seven single-file ones live
in `test_side_channel_blind_loaders.py`, which owns that guard's story.

**The VAE.** `load_state_dict_ignoring_extras` reports what it does not recognise and copies on, so
an fp8 conv weight was cast into a float32 parameter with its scale discarded -- the same shape as
the encoder's, at the four sibling VAE seams that have refused it since the seven side-channel-blind
loaders were found.

Two schemes that share the same key layout are deliberately *not* guarded here, because at these
seams they cannot pass quietly -- measured, not assumed:

- **nvfp4** packs two codes per byte, so its weight is half the width `Fp8Linear` allocates.
- **MXFP8** carries a two-dimensional E8M0 grid against a one-dimensional `weight_scale` buffer.

Both are refused by the shape check inside `load_state_dict`. An `int8_tensorwise` scale is the one
that fits: `check_int8_scale_layout` allows `[out]` as well as `[out, 1]`, and `[out]` is exactly
what that buffer expects. The `[out, 1]` spelling trips the same shape check, which is why the cell
below uses the flat one -- the other spelling would pass for the wrong reason.
"""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.ideogram4.modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer
from invokeai.backend.ideogram4.quantized_loading import (
    FP8_TEXT_ENCODER_CONFIG_FLAG,
    Fp8Linear,
    quantize_weight_to_fp8,
)
from invokeai.backend.model_manager.load.model_loaders import ideogram4
from invokeai.backend.model_manager.load.model_loaders.ideogram4 import Ideogram4DiffusersModel
from invokeai.backend.quantization.int8_convrot import CONVROT_GROUP_SIZE
from tests.fixtures.loader_seams import Seam, prepare
from tests.fixtures.quantized_payloads import comfy_quant_marker, quantize_convrot, quantize_scaled_fp8

QUANTIZED = "layers.0.feed_forward.w1"

TRANSFORMER_SEAM = Seam(
    loader=Ideogram4DiffusersModel,
    module=ideogram4,
    entry="_load_one_transformer",
    patches_device=True,
    sets_torch_dtype=False,
    casts_fp8_storage=False,
)

ENCODER_SEAM = Seam(
    loader=Ideogram4DiffusersModel,
    module=ideogram4,
    entry="_load_text_encoder",
    patches_device=True,
    sets_torch_dtype=False,
    casts_fp8_storage=False,
)


def _tiny_config() -> Ideogram4Config:
    """convrot rotates in groups along the input dim, so every Linear has to be a whole number of
    groups wide. Same module tree as the released geometry, 29 tensors instead of 458. A fresh
    instance per call: the dataclass is not frozen and it is handed to a real model."""
    return Ideogram4Config(
        emb_dim=CONVROT_GROUP_SIZE,
        num_layers=1,
        num_heads=2,
        intermediate_size=CONVROT_GROUP_SIZE,
        adanln_dim=CONVROT_GROUP_SIZE,
        in_channels=8,
        llm_features_dim=32,
        mrope_section=(16, 8, 8),
    )


def _shrink_the_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_load_one_transformer` builds `Ideogram4Config()` with no way in; the released defaults are a
    34-layer, 4608-wide model. Patched at the module the loader imports it from."""
    monkeypatch.setattr("invokeai.backend.ideogram4.modeling_ideogram4.Ideogram4Config", _tiny_config)


def _tiny_transformer_state_dict() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    reference = Ideogram4Transformer(_tiny_config())
    return {key: value.clone().to(torch.float32) for key, value in reference.state_dict().items()}


def test_an_int8_checkpoint_is_refused_rather_than_taken_for_the_private_fp8_layout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A rotated int8 weight with a flat per-output-channel scale fits `Fp8Linear` exactly: right
    weight shape, right scale shape, and a forward that applies the scale and stops there. It
    loaded, and the layer bore no relation to the stored one.
    """
    state_dict = _tiny_transformer_state_dict()
    payload = quantize_convrot(state_dict[f"{QUANTIZED}.weight"], group_size=CONVROT_GROUP_SIZE)
    state_dict[f"{QUANTIZED}.weight"] = payload.codes
    state_dict[f"{QUANTIZED}.weight_scale"] = payload.scale.squeeze(1)
    state_dict[f"{QUANTIZED}.comfy_quant"] = comfy_quant_marker(
        {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": CONVROT_GROUP_SIZE}
    )

    run = prepare(TRANSFORMER_SEAM, monkeypatch, state_dict=state_dict, geometry=_shrink_the_geometry)

    with pytest.raises(ValueError, match="quantized with convrot"):
        run.load(tmp_path)

    # `make_room` evicts other resident models to reserve space. Refusing below it would flush the
    # cache for a load that cannot finish.
    assert run.reserved == []


def test_the_other_spelling_of_the_fp8_scale_is_refused_rather_than_dropped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`.scale_weight` is the spelling ComfyUI writes, and `swap_linears_to_fp8` does not read it.
    The layer then stays an ordinary `nn.Linear`, the fp8 codes are copied in as the weight, and the
    scale is reported at DEBUG -- an effective weight off by `1/weight_scale`, measured at 7195x
    here. The int8 guard above cannot catch it, because the dtype really is fp8.
    """
    state_dict = _tiny_transformer_state_dict()
    codes, scale = quantize_weight_to_fp8(state_dict[f"{QUANTIZED}.weight"])
    state_dict[f"{QUANTIZED}.weight"] = codes
    state_dict[f"{QUANTIZED}.scale_weight"] = scale

    run = prepare(TRANSFORMER_SEAM, monkeypatch, state_dict=state_dict, geometry=_shrink_the_geometry)

    with pytest.raises(ValueError, match=r"Ideogram 4 transformer .*reads only `\.weight_scale`"):
        run.load(tmp_path)
    assert run.reserved == []


def test_an_fp8_folder_build_still_loads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The half that keeps the refusals honest: this loader's own weight-only fp8 layout goes
    through untouched, and the quantized layer arrives as an `Fp8Linear` holding its codes.
    """
    state_dict = _tiny_transformer_state_dict()
    codes, scale = quantize_weight_to_fp8(state_dict[f"{QUANTIZED}.weight"])
    state_dict[f"{QUANTIZED}.weight"] = codes
    state_dict[f"{QUANTIZED}.weight_scale"] = scale

    run = prepare(TRANSFORMER_SEAM, monkeypatch, state_dict=state_dict, geometry=_shrink_the_geometry)
    model = run.load(tmp_path)

    quantized = model.get_submodule(QUANTIZED)
    assert isinstance(quantized, Fp8Linear)
    # The codes themselves, not the dtype: `Fp8Linear.__init__` registers an empty float8 buffer, so
    # asserting the dtype alone passes on a loader that installs the layers and never fills them.
    assert torch.equal(quantized.weight.view(torch.int8), codes.view(torch.int8))
    assert torch.equal(quantized.weight_scale, scale)


#: Small enough to build eagerly. `AutoModel` gives a bare Qwen3 encoder -- the architecture family
#: the released Ideogram 4 text encoder belongs to -- so the dense cell loads a real model rather
#: than asserting against a shape the loader never reaches.
TINY_ENCODER_CONFIG = {
    "model_type": "qwen3",
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 16,
    "vocab_size": 64,
    "max_position_embeddings": 32,
    "tie_word_embeddings": False,
}

ENCODER_LAYER = "layers.0.mlp.gate_proj"


def _encoder_folder(
    tmp_path: Path, state_dict: dict[str, torch.Tensor], *, fp8_flag: bool = False, nf4: bool = False
) -> Path:
    """An Ideogram 4 model root with a `text_encoder/` component, as the loader expects one."""
    encoder = tmp_path / "text_encoder"
    encoder.mkdir(parents=True, exist_ok=True)
    config = dict(TINY_ENCODER_CONFIG)
    if fp8_flag:
        config[FP8_TEXT_ENCODER_CONFIG_FLAG] = True
    if nf4:
        config["quantization_config"] = {"quant_method": "bitsandbytes_4bit", "load_in_4bit": True}
    (encoder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    save_file(state_dict, encoder / "model.safetensors")
    return tmp_path


def _tiny_encoder_state_dict() -> dict[str, torch.Tensor]:
    from transformers import AutoConfig, AutoModel

    torch.manual_seed(0)
    config = AutoConfig.for_model(**TINY_ENCODER_CONFIG)
    return {key: value.clone() for key, value in AutoModel.from_config(config).state_dict().items()}


def test_a_quantized_text_encoder_without_the_flag_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Without the private flag this branch assigns straight into the built model, so the fp8 codes
    *become* the parameters -- off by `1/weight_scale`, 6803x for this payload -- and the orphaned
    scale is reported at DEBUG. int8 happens to raise here instead, because an int8 tensor cannot be
    a `Parameter` at all; fp8 can, which is why this is the payload that matters.
    """
    state_dict = _tiny_encoder_state_dict()
    payload = quantize_scaled_fp8(state_dict[f"{ENCODER_LAYER}.weight"].float())
    state_dict[f"{ENCODER_LAYER}.weight"] = payload.codes
    state_dict[f"{ENCODER_LAYER}.weight_scale"] = payload.scale
    root = _encoder_folder(tmp_path, state_dict)

    run = prepare(ENCODER_SEAM, monkeypatch)

    with pytest.raises(ValueError, match="quantization side channel"):
        run.load(root)
    assert run.reserved == []


def test_the_other_spelling_is_refused_in_the_flagged_branch_too(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With the flag set, the side-channel guard steps aside for the layout this branch does read --
    and `.scale_weight` is not it. Without this the flag would turn a quiet drop back on.
    """
    state_dict = _tiny_encoder_state_dict()
    codes, scale = quantize_weight_to_fp8(state_dict[f"{ENCODER_LAYER}.weight"].float())
    state_dict[f"{ENCODER_LAYER}.weight"] = codes
    state_dict[f"{ENCODER_LAYER}.scale_weight"] = scale
    root = _encoder_folder(tmp_path, state_dict, fp8_flag=True)

    run = prepare(ENCODER_SEAM, monkeypatch)

    with pytest.raises(ValueError, match=r"Ideogram 4 text encoder .*reads only `\.weight_scale`"):
        run.load(root)
    assert run.reserved == []


def test_a_qwen3_vl_text_encoder_loses_its_visual_tower(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The encoder Ideogram 4 actually ships is a Qwen3-VL, and `backend/ideogram4/text_encoding.py`
    drives its `language_model` directly -- the vision tower is never executed. It is dropped here
    both from the state dict, ahead of the reservation that is sized from it, and from the built
    module, so nothing is left on the meta device.

    The other encoder fixtures in this file are `model_type: "qwen3"`, which has no tower at all and
    therefore only exercises the tolerated case.
    """
    from transformers import Qwen3VLModel

    from tests.backend.model_manager.load.qwen3vl_gguf_fixture import tiny_qwen3vl_config

    te_config = tiny_qwen3vl_config()
    torch.manual_seed(0)
    state_dict = {k: v.contiguous() for k, v in Qwen3VLModel(te_config).state_dict().items()}

    encoder = tmp_path / "text_encoder"
    encoder.mkdir(parents=True, exist_ok=True)
    (encoder / "config.json").write_text(json.dumps(te_config.to_dict()), encoding="utf-8")
    save_file(state_dict, encoder / "model.safetensors")

    run = prepare(ENCODER_SEAM, monkeypatch)
    model = run.load(tmp_path)

    assert isinstance(model.visual, torch.nn.Identity)
    assert not any(tensor.is_meta for tensor in model.state_dict().values())
    assert any(name.startswith("language_model.layers.") for name, _ in model.named_parameters())
    # Sized from the filtered dict, so the tower is not reserved for either.
    language_bytes = sum(v.nelement() * v.element_size() for k, v in state_dict.items() if not k.startswith("visual."))
    assert run.reserved and max(run.reserved) <= language_bytes
    assert language_bytes < sum(v.nelement() * v.element_size() for v in state_dict.values())


def test_a_dense_text_encoder_still_loads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The other half, and the only end-to-end coverage this branch has: an ordinary encoder is
    built, filled, and passes the meta-device check."""
    root = _encoder_folder(tmp_path, _tiny_encoder_state_dict())

    run = prepare(ENCODER_SEAM, monkeypatch)
    model = run.load(root)

    assert not model.training
    assert not any(tensor.is_meta for tensor in model.state_dict().values())


def test_an_nf4_text_encoder_gets_past_the_guards(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """bitsandbytes sidecars are `.absmax`, `.quant_map` and `.quant_state.bitsandbytes__*`, none of
    which names a scale -- so the published nf4 encoder must reach its own branch untouched. The
    `quantization_config` is what selects that branch, and with it the loader really does build
    `InvokeLinearNF4` layers before loading; the load then fails on this fixture's shapes, which is
    not the point. What must not happen is that it fails *here*: widening `is_scale_metadata_key` to
    any bnb spelling would refuse a build that ships today, and nothing else in the tree would
    notice.

    Skipped where bitsandbytes is not installed (macOS): the branch imports it, and the broad
    `except` below would otherwise swallow that `ModuleNotFoundError` and pass having checked nothing.
    """
    pytest.importorskip("bitsandbytes")
    state_dict = _tiny_encoder_state_dict()
    weight = state_dict.pop(f"{ENCODER_LAYER}.weight")
    state_dict[f"{ENCODER_LAYER}.weight"] = torch.zeros(weight.numel() // 2, 1, dtype=torch.uint8)
    state_dict[f"{ENCODER_LAYER}.weight.absmax"] = torch.zeros(4, dtype=torch.float32)
    state_dict[f"{ENCODER_LAYER}.weight.quant_map"] = torch.zeros(16, dtype=torch.float32)
    state_dict[f"{ENCODER_LAYER}.weight.quant_state.bitsandbytes__nf4"] = torch.zeros(1, dtype=torch.uint8)
    root = _encoder_folder(tmp_path, state_dict, nf4=True)

    run = prepare(ENCODER_SEAM, monkeypatch)

    try:
        run.load(root)
    except Exception as exc:  # noqa: BLE001 - the point is *which* failure, not whether one happens
        assert "quantization side channel" not in str(exc), f"the guard fired on an nf4 encoder: {exc}"
        assert "reads only" not in str(exc), f"the spelling guard fired on an nf4 encoder: {exc}"


VAE_SEAM = Seam(
    loader=Ideogram4DiffusersModel,
    module=ideogram4,
    entry="_load_vae",
    patches_device=True,
    sets_torch_dtype=False,
    casts_fp8_storage=False,
)


def test_a_quantized_vae_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No quantized layout reaches this decoder, and nothing here looked for one.

    The payload carries the standalone `scaled_fp8` marker as well as the per-layer scale, and that
    is what pins the guard's *placement*: `convert_diffusers_state_dict` refuses a key it cannot
    rewrite, so a guard below the conversion would be pre-empted by a `KeyError` naming the marker
    instead of the scheme. The per-layer scale alone would pass either way -- the rewriter leaves it
    untouched -- so it cannot tell the two placements apart.
    """
    payload = quantize_scaled_fp8(torch.randn(4, 4))
    state_dict = {
        "encoder.conv_out.weight": payload.codes,
        "encoder.conv_out.weight_scale": payload.scale,
        "scaled_fp8": torch.ones(1),
    }

    run = prepare(VAE_SEAM, monkeypatch, state_dict=state_dict)

    with pytest.raises(ValueError, match="quantization side channel"):
        run.load(tmp_path)


def test_a_dense_vae_gets_past_the_guard(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The other half. This dict is far too small to fill a decoder, so the load fails either way --
    what must not happen is that it fails *here*."""
    run = prepare(VAE_SEAM, monkeypatch, state_dict={"encoder.conv_out.weight": torch.ones(4, 4)})

    try:
        run.load(tmp_path)
    except Exception as exc:  # noqa: BLE001 - the point is *which* failure, not whether one happens
        assert "quantization side channel" not in str(exc), f"the guard fired on a dense VAE: {exc}"
