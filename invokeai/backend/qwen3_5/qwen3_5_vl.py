"""Vendored assets for single-file Qwen3.5 vision-language models (`Qwen3_5ForConditionalGeneration`).

ComfyUI exports the FLUX.3 prompt upsampler -- a Qwen3.5-9B fine-tune -- as one safetensors file of weights: no
config, tokenizer or processor. The architecture and the generation settings are therefore vendored verbatim
(Apache-2.0) from `black-forest-labs/flux-3-image-klein-pup` at revision 3f3edb52b2c55de0d49feab28598b68f7f8e4830:

- `qwen3_5_9b_vl_config.json` is its `config.json`: `Qwen/Qwen3.5-9B`'s architecture values, re-serialized by
  transformers 5.8. Its text part equals `qwen3_5_9b_text_config.json` beside it.
- `qwen3_5_vl_generation_config.json` is its `generation_config.json`. A model built from the config alone stops
  at `<|im_end|>` only; this adds `<|endoftext|>`, as the release generates.

The tokenizer and the processor are not vendored again, because the release's are the ones already here in effect.
When this module was written, the release's own `AutoTokenizer` and `AutoProcessor` were compared with the vendored
Qwen3.5 tokenizer and `build_qwen3_vl_processor`: the same vocabulary, merges, added and special
tokens and chat template; the same ids for ASCII, CJK, emoji, combining marks, every added token and the chat
template rendered with images and `enable_thinking=False`; the same pixel values, patch grids and image-token runs
at four image sizes. The release names `Qwen3VLProcessor` itself.

transformers is imported inside the functions: model identification reads the sizes listed here at startup.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from invokeai.backend.model_manager.taxonomy import Qwen35VariantType

if TYPE_CHECKING:
    from transformers import GenerationConfig, ProcessorMixin, Qwen3_5Config

_PACKAGE_DIR = Path(__file__).parent

#: Where a `Qwen3_5ForConditionalGeneration` keeps its language model in transformers 5's key layout, which
#: ComfyUI's exports keep: the vision tower sits beside it under `model.visual.`, the LM head at `lm_head.weight`.
QWEN3_5_VL_LANGUAGE_MODEL_PREFIX = "model.language_model."

_CONFIG_FILES: dict[Qwen35VariantType, str] = {Qwen35VariantType.Qwen35_9B: "qwen3_5_9b_vl_config.json"}
_GENERATION_CONFIG_FILE = "qwen3_5_vl_generation_config.json"

#: The sizes a single-file Qwen3.5 vision-language model can be built at: those with a vendored config.
QWEN3_5_VL_VARIANTS = frozenset(_CONFIG_FILES)


def qwen3_5_vl_config(variant: Qwen35VariantType) -> "Qwen3_5Config":
    """The vendored architecture of a Qwen3.5 vision-language model of this size."""
    from transformers import Qwen3_5Config

    filename = _CONFIG_FILES.get(variant)
    if filename is None:
        raise ValueError(f"No vendored Qwen3.5 vision-language config for '{variant.value}'.")
    return Qwen3_5Config.from_dict(_read_json(filename))


def qwen3_5_vl_generation_config() -> "GenerationConfig":
    """The release's generation settings: both end-of-turn tokens and the padding id."""
    from transformers import GenerationConfig

    return GenerationConfig.from_dict(_read_json(_GENERATION_CONFIG_FILE))


def build_qwen3_5_vl_processor() -> "ProcessorMixin":
    """The release's processor: Qwen3-VL's image processor over a new instance of the vendored Qwen3.5 tokenizer.

    It renders the tokenizer's chat template, as the release's does, so a caller can apply the template through
    the processor with images in the messages. The tokenizer is its own, not the process-wide one the Qwen3.5
    encoders share (see `load_qwen3_5_tokenizer`).
    """
    from invokeai.backend.qwen3_5.qwen3_5_encoder import load_qwen3_5_tokenizer
    from invokeai.backend.qwen3_vl.qwen3_vl_assets import build_qwen3_vl_processor

    tokenizer = load_qwen3_5_tokenizer()
    processor = build_qwen3_vl_processor(tokenizer)
    processor.chat_template = tokenizer.chat_template
    return processor


def _read_json(filename: str) -> dict[str, Any]:
    raw: dict[str, Any] = json.loads((_PACKAGE_DIR / filename).read_text(encoding="utf-8"))
    return raw
