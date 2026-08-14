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

from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field

from invokeai.backend.model_manager.configs.base import Config_Base
from invokeai.backend.model_manager.configs.identification_utils import (
    NotAMatchError,
    get_config_dict_or_raise,
    raise_for_class_name,
    raise_for_override_fields,
    raise_if_not_dir,
    raise_if_not_file,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType

# PiD's caption projection is hard-wired to Gemma-2-2b's 2304-dim hidden state. Larger Gemma 2
# variants (9B → 3584, 27B → 4608) share the Gemma2ForCausalLM architecture but are incompatible.
_PID_GEMMA_HIDDEN_SIZE = 2304


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

        # Only Gemma-2-2b (2304-dim hidden state) is compatible with PiD's fixed caption projection.
        # Reject 9B/27B variants here so they are not offered as compatible encoders and then fail with
        # a matrix-shape error deep inside PiD inference.
        hidden_size = get_config_dict_or_raise(config_path).get("hidden_size")
        if hidden_size != _PID_GEMMA_HIDDEN_SIZE:
            raise NotAMatchError(
                f"Gemma2 hidden_size {hidden_size} is incompatible with PiD, which requires "
                f"{_PID_GEMMA_HIDDEN_SIZE} (Gemma-2-2b); 9B/27B variants are not supported."
            )

        # Sanity check that tokenizer files live alongside the model (PiD calls
        # AutoTokenizer.from_pretrained on the same directory).
        if not any((mod.path / f).exists() for f in ("tokenizer.json", "tokenizer.model")):
            raise NotAMatchError("directory does not contain Gemma2 tokenizer files (tokenizer.json/tokenizer.model)")

        return cls(**override_fields)


def _read_gguf_arch_and_hidden_size(path: Path) -> tuple[str, int | None]:
    """Read (general.architecture, <arch>.embedding_length) from a GGUF file's metadata.

    Raises NotAMatchError if the file is not a readable GGUF or is missing the architecture marker.
    """
    import gguf

    try:
        reader = gguf.GGUFReader(path)
    except Exception as e:
        raise NotAMatchError(f"not a readable GGUF file: {e}") from e

    arch_field = reader.fields.get("general.architecture")
    if arch_field is None:
        raise NotAMatchError("GGUF file is missing the 'general.architecture' metadata field")
    architecture = str(arch_field.contents())

    hidden_field = reader.fields.get(f"{architecture}.embedding_length")
    hidden_size = int(hidden_field.contents()) if hidden_field is not None else None
    return architecture, hidden_size


class Gemma2Encoder_GGUF_Config(Config_Base):
    """Single-file GGUF-quantized Gemma-2-2b encoder for PiD (llama.cpp GGUF, e.g. gemma-2-2b-it-Q4_K_M.gguf).

    Unlike the diffusers-directory config, this is a single ``.gguf`` file: the model config and the
    tokenizer are read from the GGUF metadata, so no companion config.json / tokenizer files are required.
    The weights are loaded natively by ``Gemma2EncoderGGUFLoader`` — the large 2D projections stay
    quantized as ``GGMLTensor`` and are dequantized on demand by the model cache, rather than being fully
    dequantized into memory at load time. Only Gemma-2-2b (2304-dim) is accepted, matching PiD's fixed
    caption projection; 9B/27B GGUFs are rejected here as for the directory config.
    """

    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.Gemma2Encoder] = Field(default=ModelType.Gemma2Encoder)
    format: Literal[ModelFormat.GGUFQuantized] = Field(default=ModelFormat.GGUFQuantized)
    cpu_only: bool | None = Field(default=None, description="Whether this model should run on CPU only")

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)
        raise_for_override_fields(cls, override_fields)

        if mod.path.suffix.lower() != ".gguf":
            raise NotAMatchError(f"not a .gguf file: {mod.path.name}")

        architecture, hidden_size = _read_gguf_arch_and_hidden_size(mod.path)
        if architecture != "gemma2":
            raise NotAMatchError(f"GGUF architecture '{architecture}' is not 'gemma2'")
        if hidden_size != _PID_GEMMA_HIDDEN_SIZE:
            raise NotAMatchError(
                f"Gemma2 GGUF embedding_length {hidden_size} is incompatible with PiD, which requires "
                f"{_PID_GEMMA_HIDDEN_SIZE} (Gemma-2-2b); 9B/27B variants are not supported."
            )

        return cls(**override_fields)
