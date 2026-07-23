"""Model config for the Gemma-2-2b-it text encoder used by PiD.

PiD's pre-trained decoders condition on Gemma-2-2b-it caption embeddings
(2304-dim). This config recognises a stand-alone diffusers/transformers
directory containing a Gemma2 causal LM (config.json + safetensors weights +
tokenizer files).

The reference model PiD uses is `Efficient-Large-Model/gemma-2-2b-it`, an
ungated mirror of `google/gemma-2-2b-it`. Both produce a
`Gemma2ForCausalLM` config which is what we match on.

License note: Gemma 2 is distributed under the Gemma Terms of Use (Google).
This config only describes how to recognise the model on disk; downloading
and accepting Gemma's license is the user's responsibility.
"""

from typing import Any, Literal, Self

from pydantic import Field

from invokeai.backend.model_manager.configs.base import Config_Base
from invokeai.backend.model_manager.configs.identification_utils import (
    NotAMatchError,
    raise_for_class_name,
    raise_for_override_fields,
    raise_if_not_dir,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType


class Gemma2Encoder_Gemma2Encoder_Config(Config_Base):
    """Standalone Gemma-2 causal LM directory used as a text encoder by PiD.

    Expected directory layout (HuggingFace `from_pretrained`-compatible)::

        <model_root>/
            config.json             # architectures: ["Gemma2ForCausalLM"]
            tokenizer.json
            tokenizer_config.json
            model-*.safetensors     # or model.safetensors / *.bin
    """

    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.Gemma2Encoder] = Field(default=ModelType.Gemma2Encoder)
    format: Literal[ModelFormat.Gemma2Encoder] = Field(default=ModelFormat.Gemma2Encoder)
    cpu_only: bool | None = Field(default=None, description="Whether this model should run on CPU only")

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)
        raise_for_override_fields(cls, override_fields)

        config_path = mod.path / "config.json"
        if not config_path.exists():
            raise NotAMatchError(f"missing config.json at {config_path}")

        # Reject full diffusers pipelines (they have model_index.json at root).
        if (mod.path / "model_index.json").exists():
            raise NotAMatchError("directory looks like a full diffusers pipeline, not a standalone Gemma2 encoder")

        # Architecture marker is the canonical signal.
        raise_for_class_name(config_path, {"Gemma2ForCausalLM"})

        # Sanity check that tokenizer files live alongside the model (PiD calls
        # AutoTokenizer.from_pretrained on the same directory).
        if not any((mod.path / f).exists() for f in ("tokenizer.json", "tokenizer.model")):
            raise NotAMatchError("directory does not contain Gemma2 tokenizer files (tokenizer.json/tokenizer.model)")

        return cls(**override_fields)
