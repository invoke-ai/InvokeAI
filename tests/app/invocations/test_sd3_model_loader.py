"""Where the SD 3 model loader sources each part from.

A Diffusers main brings everything. Inside its pipeline CLIP-G is the `text_encoder_2` submodel; a CLIP-G installed on
its own is a model whose encoder is its `TextEncoder`, and the CLIP loaders know no other, so asking it for
`TextEncoder2` failed every generation that selected one.

A GGUF main holds only the transformer, so every other part, each tokenizer included, comes from its component.
"""

from types import SimpleNamespace

import pytest

from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.invocations.sd3.sd3_model_loader import Sd3ModelLoaderInvocation
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType, SubModelType


def _identifier(name: str, base: BaseModelType, model_type: ModelType) -> ModelIdentifierField:
    return ModelIdentifierField(key=name, hash=f"blake3:{name}", name=name, base=base, type=model_type)


def _invoke(main_format: ModelFormat = ModelFormat.Diffusers, **components: ModelIdentifierField):
    main = _identifier("sd3.5-medium", BaseModelType.StableDiffusion3, ModelType.Main)
    invocation = Sd3ModelLoaderInvocation(id="loader", model=main, **components)
    configs = {"sd3.5-medium": SimpleNamespace(format=main_format)}
    context = SimpleNamespace(models=SimpleNamespace(get_config=lambda model: configs[model.key]))
    return invocation.invoke(context)  # type: ignore[arg-type]


COMPONENTS = {
    "t5_encoder_model": _identifier("t5", BaseModelType.Any, ModelType.T5Encoder),
    "clip_l_model": _identifier("clip_l", BaseModelType.Any, ModelType.CLIPEmbed),
    "clip_g_model": _identifier("clip_g", BaseModelType.Any, ModelType.CLIPEmbed),
    "vae_model": _identifier("sd3_vae", BaseModelType.StableDiffusion3, ModelType.VAE),
}


def test_the_pipelines_own_clip_g_is_its_second_text_encoder() -> None:
    output = _invoke()

    assert output.clip_g.text_encoder.submodel_type is SubModelType.TextEncoder2
    assert output.clip_g.tokenizer.submodel_type is SubModelType.Tokenizer2


def test_a_separately_selected_clip_g_is_asked_for_its_own_text_encoder() -> None:
    output = _invoke(clip_g_model=COMPONENTS["clip_g_model"])

    assert output.clip_g.text_encoder.key == "clip_g"
    assert output.clip_g.text_encoder.submodel_type is SubModelType.TextEncoder
    # A Diffusers main tokenizes with the pipeline's own CLIP-G tokenizer either way.
    assert output.clip_g.tokenizer.key == "sd3.5-medium"


def test_a_gguf_main_takes_every_other_part_from_its_components() -> None:
    output = _invoke(ModelFormat.GGUFQuantized, **COMPONENTS)

    def source(field: ModelIdentifierField) -> tuple[str, SubModelType | None]:
        return field.key, field.submodel_type

    assert source(output.transformer.transformer) == ("sd3.5-medium", SubModelType.Transformer)
    assert source(output.clip_l.tokenizer) == ("clip_l", SubModelType.Tokenizer)
    assert source(output.clip_l.text_encoder) == ("clip_l", SubModelType.TextEncoder)
    assert source(output.clip_g.tokenizer) == ("clip_g", SubModelType.Tokenizer)
    assert source(output.clip_g.text_encoder) == ("clip_g", SubModelType.TextEncoder)
    # A standalone T5 holds its parts in the slots FLUX's pipeline uses.
    assert source(output.t5_encoder.tokenizer) == ("t5", SubModelType.Tokenizer2)
    assert source(output.t5_encoder.text_encoder) == ("t5", SubModelType.TextEncoder2)
    assert source(output.vae.vae) == ("sd3_vae", SubModelType.VAE)


def test_a_gguf_main_without_its_components_names_every_missing_one() -> None:
    with pytest.raises(
        ValueError, match=r"its CLIP L Encoder, CLIP G Encoder and VAE must be selected on Main Model - SD3"
    ):
        _invoke(ModelFormat.GGUFQuantized, t5_encoder_model=COMPONENTS["t5_encoder_model"])
