"""Model config for the LTX-2.5 Gemma-4 text encoder.

LTX-2.5 conditions on a Lightricks fine-tune of Gemma-4-12B (``gemma4-12b-ltx-v1``): every one of
its 49 hidden states (48 layers + the embedding output) is stacked into a 188160-wide vector per
token and projected by the LTX-2 text connectors. Google's stock Gemma-4 is *not* interchangeable
with it, and only the text tower is used -- the vision and audio embedders of the unified model are
never loaded.

The folder is the HF-style layout the ungated ``DeepBeepMeep/LTX-2`` mirror ships under
``gemma4-12b-ltx-v1/``::

    <model_root>/
        config.json                 # model_type gemma4_unified_text / architectures Gemma4UnifiedForCausalLM
        tokenizer.json
        tokenizer_config.json
        chat_template.jinja
        gemma4-12b-ltx-v1_bf16.safetensors  OR  gemma4-12b-ltx-v1_int8_convrot.safetensors

When installed from a Hugging Face source that lists the files explicitly (the starter model
does, to fetch one weight file rather than the mirror's whole 37 GB folder), the installer keeps
their repo-relative paths, so the same files sit one directory down::

    <model_root>/gemma4-12b-ltx-v1/config.json ...

``subfolder`` records which of the two it is.

The weights carry the text tower's ``model.*`` keys only (no ``lm_head``: the head is tied and never
run). The text tower is loaded into ``transformers.Gemma4TextModel`` -- verified key-for-key against
the checkpoint -- so the installed transformers needs no ``Gemma4Unified*`` classes.

A GGUF build of the same tower is one file (``Gemma4Encoder_GGUF_LTX2_Config``): the community
"with-proj" conversions of the release (elix3r, Rootport) keep its Hugging Face tensor names and pack
the tokenizer and the model config in beside the weights -- the tokenizer files as byte tensors, the
unified config as a metadata string. They also carry the text projection and the vision/audio stubs
of the unified model, which nothing here reads.

License note: the weights are derivatives of the official LTX-2.5 release and are distributed under
the LTX-2.x Community License. This config only describes how to recognise the folder on disk.
"""

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field

from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base, Config_Base
from invokeai.backend.model_manager.configs.identification_utils import (
    InvalidMatchError,
    NotAMatchError,
    get_config_dict_or_raise,
    raise_for_override_fields,
    raise_if_not_dir,
    raise_if_not_file,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor

# The LTX-2.5 connectors project ``hidden_size * (num_hidden_layers + 1)`` = 3840 * 49 = 188160
# features; a Gemma-4 of any other width or depth cannot feed them.
LTX2_GEMMA4_HIDDEN_SIZE = 3840
LTX2_GEMMA4_NUM_LAYERS = 48

_GEMMA4_TEXT_MODEL_TYPES = {"gemma4_unified_text", "gemma4_text"}
_GEMMA4_TEXT_ARCHITECTURES = {"Gemma4UnifiedForCausalLM", "Gemma4ForCausalLM", "Gemma4TextModel"}

# Where the GGUF conversions keep the unified model's config.json: elix3r under the first key, Rootport
# under the second (which the LTX-2 transformer GGUFs use for the transformer config, without a
# ``text_config``).
GGUF_CONFIG_KEYS = ("source.safetensors.gemma_config", "config")
# The tokenizer files a GGUF conversion packs in, one byte per element: the two
# ``AutoTokenizer.from_pretrained`` needs to rebuild the folder's tokenizer, pad and EOS included.
GGUF_TOKENIZER_ASSETS = {"tokenizer_json": "tokenizer.json", "hf_asset__tokenizer_config.json": "tokenizer_config.json"}


def _find_model_subfolder(model_root: Path) -> str:
    """'' when config.json is at the root, else the one immediate subdirectory that holds it."""
    if (model_root / "config.json").is_file():
        return ""
    candidates = sorted(p for p in model_root.iterdir() if p.is_dir() and (p / "config.json").is_file())
    if len(candidates) != 1:
        raise NotAMatchError("no config.json at the folder root or in exactly one subdirectory")
    return candidates[0].name


def select_gemma4_weight_file(folder: Path) -> Path:
    """The one weight file a Gemma-4 encoder folder is loaded from.

    The LTX release is a single file per precision. A sharded folder (``model-00001-of-00005`` plus
    an index) is Google's stock Gemma-4 layout, which this loader does not read and whose weights
    are not the LTX fine-tune anyway; leaving it unclaimed keeps it a generic text LLM. The mirror
    publishes the bf16 and int8-convrot files side by side, so a user who downloads both gets a
    deterministic pick: the bf16 file, the lossless one.
    """
    if (folder / "model.safetensors.index.json").is_file():
        raise NotAMatchError("a sharded Gemma-4 folder is not the single-file LTX-2 text encoder")
    candidates = sorted(p for p in folder.glob("*.safetensors") if p.is_file())
    if not candidates:
        raise NotAMatchError("no .safetensors weight file in the folder")
    if any(re.search(r"-\d{5}-of-\d{5}\.safetensors$", p.name) for p in candidates):
        raise NotAMatchError("a sharded Gemma-4 folder is not the single-file LTX-2 text encoder")
    preferred = [p for p in candidates if "bf16" in p.name.lower()]
    return (preferred or candidates)[0]


def is_ltx2_gemma4_encoder_folder(model_root: Path) -> bool:
    """Whether ``Gemma4Encoder_Gemma4Encoder_LTX2_Config`` would claim this folder.

    The generic text-LLM config asks this to know when to stand aside: only for the LTX layout
    (the Gemma-4-12B text config beside a single unsharded weight file), never for stock Gemma-4.
    """
    try:
        _resolve_ltx2_gemma4_layout(model_root)
    except NotAMatchError:
        return False
    return True


def _resolve_ltx2_gemma4_layout(model_root: Path, subfolder: str | None = None) -> tuple[str, Path]:
    """``(subfolder, weight_file)`` for a folder in the LTX Gemma-4 layout, or ``NotAMatchError``."""
    if subfolder is None:
        subfolder = _find_model_subfolder(model_root)
    root = model_root / subfolder if subfolder else model_root

    config = get_config_dict_or_raise(root / "config.json")
    # Top-level only: a unified multimodal config nests a text config this loader could build, but
    # its weights carry the vision/audio embedders too and are not the LTX fine-tune.
    model_type = config.get("model_type")
    architectures = set(config.get("architectures") or [])
    if model_type not in _GEMMA4_TEXT_MODEL_TYPES and not (architectures & _GEMMA4_TEXT_ARCHITECTURES):
        raise NotAMatchError(f"config.json does not describe a Gemma-4 text model (model_type={model_type!r})")

    hidden_size = config.get("hidden_size")
    num_layers = config.get("num_hidden_layers")
    if hidden_size != LTX2_GEMMA4_HIDDEN_SIZE or num_layers != LTX2_GEMMA4_NUM_LAYERS:
        raise NotAMatchError(
            f"LTX-2.5 needs the Gemma-4-12B tower (hidden_size {LTX2_GEMMA4_HIDDEN_SIZE}, "
            f"{LTX2_GEMMA4_NUM_LAYERS} layers); this folder has hidden_size {hidden_size}, {num_layers} layers"
        )
    if not (root / "tokenizer.json").exists():
        raise NotAMatchError("no tokenizer.json beside config.json")
    return subfolder, select_gemma4_weight_file(root)


class Gemma4Encoder_Gemma4Encoder_LTX2_Config(Config_Base):
    """The LTX-2.5 Gemma-4-12B text encoder as a folder (config, tokenizer, one weight file)."""

    base: Literal[BaseModelType.LTX2] = Field(default=BaseModelType.LTX2)
    type: Literal[ModelType.Gemma4Encoder] = Field(default=ModelType.Gemma4Encoder)
    format: Literal[ModelFormat.Gemma4Encoder] = Field(default=ModelFormat.Gemma4Encoder)
    subfolder: str = Field(
        default="",
        description="Directory inside the model folder that holds config, tokenizer and weights ('' = the root).",
    )
    weight_file: str = Field(description="The weight file inside `subfolder` the encoder loads from.")
    cpu_only: bool | None = Field(default=None, description="Whether this model should run on CPU only")

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)
        raise_for_override_fields(cls, override_fields)

        subfolder, selected = _resolve_ltx2_gemma4_layout(mod.path, override_fields.pop("subfolder", None))
        root = mod.path / subfolder if subfolder else mod.path

        weight_file = override_fields.pop("weight_file", None) or selected.name
        if not (root / weight_file).is_file():
            raise NotAMatchError(f"weight file {weight_file!r} is not in the folder")

        return cls(**override_fields, subfolder=subfolder, weight_file=weight_file)


def gguf_gemma4_text_config(metadata: Mapping[str, str]) -> dict[str, Any]:
    """The Gemma-4 text config a GGUF conversion embeds in its metadata, or ``NotAMatchError``."""
    for key in GGUF_CONFIG_KEYS:
        try:
            config = json.loads(metadata[key])
        except (KeyError, ValueError):
            continue
        text_config = config.get("text_config", config) if isinstance(config, dict) else None
        if isinstance(text_config, dict) and {"hidden_size", "num_hidden_layers"} <= text_config.keys():
            return text_config
    raise NotAMatchError("the GGUF metadata carries no Gemma-4 text config")


def _layer_count(state_dict: Mapping[Any, Any], prefix: str) -> int:
    indices = {int(m.group(1)) for key in state_dict if (m := re.match(rf"{re.escape(prefix)}(\d+)\.", str(key)))}
    return max(indices, default=-1) + 1


class Gemma4Encoder_GGUF_LTX2_Config(Checkpoint_Config_Base, Config_Base):
    """The LTX-2.5 Gemma-4-12B text encoder as one GGUF file, with its tokenizer and config packed in."""

    base: Literal[BaseModelType.LTX2] = Field(default=BaseModelType.LTX2)
    type: Literal[ModelType.Gemma4Encoder] = Field(default=ModelType.Gemma4Encoder)
    format: Literal[ModelFormat.GGUFQuantized] = Field(default=ModelFormat.GGUFQuantized)
    cpu_only: bool | None = Field(default=None, description="Whether this model should run on CPU only")

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)
        raise_for_override_fields(cls, override_fields)
        if mod.path.suffix != ".gguf":
            raise NotAMatchError("not a GGUF file")

        # Recognised by the tower, not the header: elix3r's builds name the architecture `gemma4`,
        # Rootport's `ltxv`.
        sd = mod.load_state_dict()
        embedding = sd.get("model.embed_tokens.weight")
        if embedding is None:
            # Google's Gemma-4-12B as llama.cpp converts it. Named from the header: Gemma-3-12B (LTX-2.0's
            # encoder) has the same width and depth, and is no business of this config.
            stock = sd.get("token_embd.weight")
            if (
                stock is not None
                and stock.shape[-1] == LTX2_GEMMA4_HIDDEN_SIZE
                and _layer_count(sd, "blk.") == LTX2_GEMMA4_NUM_LAYERS
                and mod.metadata().get("general.architecture") == "gemma4"
            ):
                raise InvalidMatchError(
                    f"{mod.path.name} is Google's Gemma-4-12B in the llama.cpp layout, not the LTX-tuned encoder "
                    "LTX-2.5 conditions on. Use an LTX-2.5 text-encoder GGUF (e.g. elix3r or Rootport) or the "
                    "LTX-2.5 Text Encoder starter model."
                )
            raise NotAMatchError("no Gemma tower in Hugging Face naming")
        if "model.layers.0.layer_scalar" not in sd:
            raise NotAMatchError("no Gemma-4 layer scalars")
        if (
            embedding.shape[-1] != LTX2_GEMMA4_HIDDEN_SIZE
            or _layer_count(sd, "model.layers.") != LTX2_GEMMA4_NUM_LAYERS
        ):
            raise NotAMatchError("not the Gemma-4-12B tower the LTX-2.5 connectors read")
        if not any(isinstance(value, GGMLTensor) for value in sd.values()):
            raise NotAMatchError("no GGUF tensors")

        try:
            text_config = gguf_gemma4_text_config(mod.metadata())
        except NotAMatchError:
            text_config = {}
        if (text_config.get("hidden_size"), text_config.get("num_hidden_layers")) != (
            LTX2_GEMMA4_HIDDEN_SIZE,
            LTX2_GEMMA4_NUM_LAYERS,
        ):
            raise InvalidMatchError(
                f"{mod.path.name} holds the LTX-2.5 Gemma-4 tower but no model config for it in its metadata, so "
                "it cannot be built. Use an LTX-2.5 text-encoder GGUF that embeds one (e.g. elix3r or Rootport)."
            )
        if missing := [name for name in GGUF_TOKENIZER_ASSETS if name not in sd]:
            raise InvalidMatchError(
                f"{mod.path.name} holds the LTX-2.5 Gemma-4 tower but carries no tokenizer (missing {missing}). "
                "Use an LTX-2.5 text-encoder GGUF that packs its tokenizer in (e.g. elix3r or Rootport)."
            )
        return cls(**override_fields)
