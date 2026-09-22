"""The FLUX.1 model loader must accept a complete SDNQ pipeline on its own.

`docs/.../sdnq-quantization.mdx` promises "one install pulls everything you need (transformer +
encoders + VAE)", but the node required separate T5, CLIP and VAE identifiers regardless, forcing
users to install duplicates of components the pipeline already shipped. Models that genuinely cannot
supply the parts (single-file / GGUF / BnB) must still say so, and say which parts are missing.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from invokeai.app.invocations.flux_model_loader import FluxModelLoaderInvocation
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.backend.model_manager.configs.main import Main_SDNQ_Diffusers_FLUX_Config
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    FluxVariantType,
    ModelFormat,
    ModelType,
    SubModelType,
)

_FLUX1_PIPELINE_SUBMODELS = {
    SubModelType.Transformer: object(),
    SubModelType.VAE: object(),
    SubModelType.TextEncoder: object(),
    SubModelType.Tokenizer: object(),
    SubModelType.TextEncoder2: object(),
    SubModelType.Tokenizer2: object(),
}


def _identifier(key: str, model_type: ModelType = ModelType.Main) -> ModelIdentifierField:
    return ModelIdentifierField(key=key, hash=f"hash:{key}", name=key, base=BaseModelType.Flux, type=model_type)


def _context(config) -> MagicMock:
    context = MagicMock()
    context.models.exists.return_value = True
    context.models.get_config.return_value = config
    return context


def _sdnq_pipeline(submodels: dict | None = None) -> Main_SDNQ_Diffusers_FLUX_Config:
    # A real config object, because the invocation asserts on its type before reading `variant`.
    return Main_SDNQ_Diffusers_FLUX_Config.model_construct(
        format=ModelFormat.SDNQQuantized,
        submodels=_FLUX1_PIPELINE_SUBMODELS if submodels is None else submodels,
        variant=FluxVariantType.Dev,
    )


def _single_file() -> SimpleNamespace:
    return SimpleNamespace(format=ModelFormat.Checkpoint, submodels=None, variant="dev")


def _invoke(main, **components):
    invocation = FluxModelLoaderInvocation.model_construct(
        model=main,
        t5_encoder_model=components.get("t5"),
        clip_embed_model=components.get("clip"),
        vae_model=components.get("vae"),
    )
    return invocation


def test_a_complete_sdnq_pipeline_supplies_every_component_itself() -> None:
    main = _identifier("sdnq-pipeline")

    output = _invoke(main).invoke(_context(_sdnq_pipeline()))

    # Every part resolves back to the pipeline install, at the right submodel slot.
    assert output.vae.vae.key == "sdnq-pipeline"
    assert output.vae.vae.submodel_type is SubModelType.VAE
    assert output.clip.tokenizer.submodel_type is SubModelType.Tokenizer
    assert output.clip.text_encoder.submodel_type is SubModelType.TextEncoder
    assert output.t5_encoder.tokenizer.submodel_type is SubModelType.Tokenizer2
    assert output.t5_encoder.text_encoder.submodel_type is SubModelType.TextEncoder2
    assert output.transformer.transformer.submodel_type is SubModelType.Transformer


def test_an_explicit_selection_still_wins_over_the_pipeline() -> None:
    """Supplying a component explicitly must keep overriding the bundled one."""
    main = _identifier("sdnq-pipeline")
    external_vae = _identifier("my-vae", ModelType.VAE)

    output = _invoke(main, vae=external_vae).invoke(_context(_sdnq_pipeline()))

    assert output.vae.vae.key == "my-vae"
    # ...while the parts left unset still come from the pipeline.
    assert output.clip.tokenizer.key == "sdnq-pipeline"


def test_a_single_file_model_without_components_names_all_the_missing_parts() -> None:
    """Models that cannot supply the parts must still be rejected, and say what to select."""
    invocation = _invoke(_identifier("flux-gguf"))

    with pytest.raises(ValueError) as excinfo:
        invocation.invoke(_context(_single_file()))

    message = str(excinfo.value)
    assert "T5 Encoder" in message and "CLIP Embed" in message and "VAE" in message


def test_an_incomplete_sdnq_pipeline_without_t5_still_requires_one() -> None:
    """A FLUX.1 pipeline needs two encoders; missing the T5 pair is not self-contained."""
    partial = {
        k: v
        for k, v in _FLUX1_PIPELINE_SUBMODELS.items()
        if k not in (SubModelType.TextEncoder2, SubModelType.Tokenizer2)
    }
    invocation = _invoke(_identifier("sdnq-no-t5"))

    with pytest.raises(ValueError) as excinfo:
        invocation.invoke(_context(_sdnq_pipeline(partial)))

    assert "T5 Encoder" in str(excinfo.value)
