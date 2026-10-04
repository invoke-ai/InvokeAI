"""What the Ideogram 4 loader node refuses to hand to a generation.

Two of these are not ordinary type checks. The branch check rejects a *swapped* pair, which would
otherwise run happily and produce images that ignore the prompt; the encoder check rejects Krea-2's
Qwen3-VL 4B, which installs under the same model type and would fail as a shape mismatch inside the
first denoising step.
"""

from types import SimpleNamespace

import pytest

from invokeai.app.invocations.ideogram4.ideogram4_model_loader import Ideogram4ModelLoaderInvocation
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelType,
    Qwen3VLVariantType,
    SubModelType,
)


def _model(key: str, base: BaseModelType, model_type: ModelType) -> ModelIdentifierField:
    return ModelIdentifierField(key=key, hash=f"hash-{key}", name=key, base=base, type=model_type)


def _context(configs: dict[str, SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(
        logger=SimpleNamespace(warning=lambda *_args, **_kwargs: None),
        models=SimpleNamespace(get_config=lambda identifier: configs[identifier.key]),
    )


def _main(model_format: ModelFormat, branch: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        base=BaseModelType.Ideogram4,
        branch=branch,
        format=model_format,
        name=f"ideogram4-{branch or model_format.value}",
        type=ModelType.Main,
    )


def _encoder(variant: Qwen3VLVariantType) -> SimpleNamespace:
    return SimpleNamespace(
        base=BaseModelType.Any,
        format=ModelFormat.Checkpoint,
        name=f"qwen3-vl-{variant.value}",
        type=ModelType.Qwen3VLEncoder,
        variant=variant,
    )


def _vae(base: BaseModelType = BaseModelType.Flux2) -> SimpleNamespace:
    return SimpleNamespace(base=base, format=ModelFormat.Checkpoint, name=f"{base.value}-vae", type=ModelType.VAE)


CONDITIONAL = _model("cond", BaseModelType.Ideogram4, ModelType.Main)
UNCONDITIONAL = _model("uncond", BaseModelType.Ideogram4, ModelType.Main)
ENCODER = _model("encoder", BaseModelType.Any, ModelType.Qwen3VLEncoder)
VAE = _model("vae", BaseModelType.Flux2, ModelType.VAE)


def _single_file_configs(**overrides: SimpleNamespace) -> dict[str, SimpleNamespace]:
    configs = {
        "cond": _main(ModelFormat.Checkpoint, "conditional"),
        "uncond": _main(ModelFormat.Checkpoint, "unconditional"),
        "encoder": _encoder(Qwen3VLVariantType.Qwen3VL_8B),
        "vae": _vae(),
    }
    configs.update(overrides)
    return configs


def _invoke(configs: dict[str, SimpleNamespace], **fields):
    node = Ideogram4ModelLoaderInvocation(
        model=CONDITIONAL,
        unconditional_model=UNCONDITIONAL,
        qwen3_vl_encoder_model=ENCODER,
        vae_model=VAE,
        **fields,
    )
    return node.invoke(_context(configs))


def test_a_single_file_pair_emits_both_branches_and_the_selected_components() -> None:
    output = _invoke(_single_file_configs())

    assert output.transformer.transformer.key == "cond"
    assert output.transformer.transformer.submodel_type is SubModelType.Transformer
    assert output.unconditional_transformer is not None
    assert output.unconditional_transformer.transformer.key == "uncond"
    assert output.unconditional_transformer.transformer.submodel_type is SubModelType.Transformer
    assert output.qwen3_encoder.text_encoder.key == "encoder"
    assert output.vae.vae.key == "vae"


def test_a_diffusers_pipeline_serves_every_submodel_from_itself() -> None:
    configs = {"cond": _main(ModelFormat.Diffusers)}
    output = Ideogram4ModelLoaderInvocation(model=CONDITIONAL).invoke(_context(configs))

    assert output.unconditional_transformer is None
    assert output.qwen3_encoder.tokenizer.key == "cond"
    assert output.qwen3_encoder.text_encoder.submodel_type is SubModelType.TextEncoder
    assert output.vae.vae.key == "cond"


def test_a_swapped_pair_is_refused() -> None:
    configs = _single_file_configs(
        cond=_main(ModelFormat.Checkpoint, "unconditional"),
        uncond=_main(ModelFormat.Checkpoint, "conditional"),
    )

    # The slot is the actionable half: "conditional branch" alone is a substring of "unconditional
    # branch", so without naming the slot this test would pass on either message.
    with pytest.raises(ValueError, match="is the unconditional branch.*as 'Model'"):
        _invoke(configs)


def test_the_unconditional_slot_refuses_a_second_conditional_file() -> None:
    configs = _single_file_configs(uncond=_main(ModelFormat.Checkpoint, "conditional"))

    with pytest.raises(ValueError, match=r"is the conditional branch.*Transformer \(Unconditional\)"):
        _invoke(configs)


def test_the_same_file_cannot_serve_both_branches() -> None:
    node = Ideogram4ModelLoaderInvocation(
        model=CONDITIONAL,
        unconditional_model=CONDITIONAL,
        qwen3_vl_encoder_model=ENCODER,
        vae_model=VAE,
    )

    with pytest.raises(ValueError, match="two different files"):
        node.invoke(_context(_single_file_configs()))


@pytest.mark.parametrize(
    ("missing", "match"),
    [
        ("unconditional_model", "one of two branches"),
        ("qwen3_vl_encoder_model", "no text encoder"),
        ("vae_model", "no VAE"),
    ],
)
def test_a_single_file_main_needs_every_component(missing: str, match: str) -> None:
    fields = {"unconditional_model": UNCONDITIONAL, "qwen3_vl_encoder_model": ENCODER, "vae_model": VAE}
    fields[missing] = None

    node = Ideogram4ModelLoaderInvocation(model=CONDITIONAL, **fields)

    with pytest.raises(ValueError, match=match):
        node.invoke(_context(_single_file_configs()))


def test_krea2s_qwen3_vl_encoder_is_refused() -> None:
    configs = _single_file_configs(encoder=_encoder(Qwen3VLVariantType.Qwen3VL_4B))

    with pytest.raises(ValueError, match="is the Qwen3-VL 4B encoder"):
        _invoke(configs)


def test_an_encoder_with_no_recorded_size_is_refused_readably() -> None:
    """MiniMax H3's truncated Qwen3-VL-32B shares this model type and carries no variant at all.

    The picker no longer offers it, but a hand-built graph still can, and "the Qwen3-VL None
    encoder" is not a sentence that helps anyone.
    """
    encoder = SimpleNamespace(
        base=BaseModelType.MiniMaxH3,
        format=ModelFormat.Checkpoint,
        name="qwen3vl_32b_minimax_h3_fp8",
        type=ModelType.Qwen3VLEncoder,
    )
    configs = _single_file_configs(encoder=encoder)

    with pytest.raises(ValueError, match="of an unrecorded size"):
        _invoke(configs)


def test_the_encoder_picker_excludes_another_architectures_qwen3_vl() -> None:
    # MiniMax H3's is base `minimax-h3`; the two this node can use are base-agnostic components.
    field = Ideogram4ModelLoaderInvocation.model_fields["qwen3_vl_encoder_model"]

    assert field.json_schema_extra is not None
    assert field.json_schema_extra["ui_model_base"] == [BaseModelType.Any.value]


def test_a_vae_from_another_family_is_refused() -> None:
    configs = _single_file_configs(vae=_vae(BaseModelType.StableDiffusionXL))

    with pytest.raises(ValueError, match="not compatible with Ideogram 4"):
        _invoke(configs)


def test_a_diffusers_model_cannot_be_the_unconditional_branch() -> None:
    configs = _single_file_configs(uncond=_main(ModelFormat.Diffusers, "unconditional"))

    with pytest.raises(ValueError, match="must be a single-file"):
        _invoke(configs)


def test_the_vae_picker_offers_the_bases_the_architecture_accepts() -> None:
    field = Ideogram4ModelLoaderInvocation.model_fields["vae_model"]

    assert field.json_schema_extra is not None
    assert set(field.json_schema_extra["ui_model_base"]) == {BaseModelType.Flux2.value}
