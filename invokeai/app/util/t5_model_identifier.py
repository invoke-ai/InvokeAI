from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.backend.model_manager.taxonomy import BaseModelType, SubModelType


def preprocess_t5_encoder_model_identifier(model_identifier: ModelIdentifierField) -> ModelIdentifierField:
    """A helper function to normalize a T5 encoder model identifier so that T5 models associated with FLUX
    or SD3 models can be used interchangeably.
    """
    if model_identifier.base == BaseModelType.Any:
        return model_identifier.model_copy(update={"submodel_type": SubModelType.TextEncoder2})
    elif model_identifier.base == BaseModelType.StableDiffusion3:
        return model_identifier.model_copy(update={"submodel_type": SubModelType.TextEncoder3})
    else:
        raise ValueError(f"Unsupported model base: {model_identifier.base}")


def preprocess_t5_tokenizer_model_identifier(model_identifier: ModelIdentifierField) -> ModelIdentifierField:
    """A helper function to normalize a T5 tokenizer model identifier so that T5 models associated with FLUX
    or SD3 models can be used interchangeably.
    """
    if model_identifier.base == BaseModelType.Any:
        return model_identifier.model_copy(update={"submodel_type": SubModelType.Tokenizer2})
    elif model_identifier.base == BaseModelType.StableDiffusion3:
        return model_identifier.model_copy(update={"submodel_type": SubModelType.Tokenizer3})
    else:
        raise ValueError(f"Unsupported model base: {model_identifier.base}")
