"""Which encoder the SD 3 model loader asks a separately selected CLIP for.

Inside the SD 3 pipeline CLIP-G is the `text_encoder_2` submodel. A CLIP-G installed on its own is a model whose
encoder is its `TextEncoder`, and the CLIP loaders know no other; asking it for `TextEncoder2` failed every
generation that selected one.
"""

from types import SimpleNamespace

from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.invocations.sd3.sd3_model_loader import Sd3ModelLoaderInvocation
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType, SubModelType


def _identifier(name: str, base: BaseModelType, model_type: ModelType) -> ModelIdentifierField:
    return ModelIdentifierField(key=name, hash=f"blake3:{name}", name=name, base=base, type=model_type)


def _invoke(**components: ModelIdentifierField):
    main = _identifier("sd3.5-medium", BaseModelType.StableDiffusion3, ModelType.Main)
    invocation = Sd3ModelLoaderInvocation(id="loader", model=main, **components)
    return invocation.invoke(SimpleNamespace())  # type: ignore[arg-type]


def test_the_pipelines_own_clip_g_is_its_second_text_encoder() -> None:
    output = _invoke()

    assert output.clip_g.text_encoder.submodel_type is SubModelType.TextEncoder2
    assert output.clip_g.tokenizer.submodel_type is SubModelType.Tokenizer2


def test_a_separately_selected_clip_g_is_asked_for_its_own_text_encoder() -> None:
    clip_g = _identifier("clip_g", BaseModelType.Any, ModelType.CLIPEmbed)

    output = _invoke(clip_g_model=clip_g)

    assert output.clip_g.text_encoder.key == "clip_g"
    assert output.clip_g.text_encoder.submodel_type is SubModelType.TextEncoder
    # SD 3 tokenizes with the pipeline's own CLIP-G tokenizer either way.
    assert output.clip_g.tokenizer.key == "sd3.5-medium"
