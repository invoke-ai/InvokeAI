"""Bundled tokenizer and architectures for CLIP-L and CLIP-G text encoders installed as single files.

ComfyUI distributes the text towers SD 3 and FLUX.1 use as bare state dicts (``clip_l``, ``clip_g``) with no
config and no tokenizer. Both are fixed architectures, so their configs are vendored here rather than inferred:
CLIP-L as ``openai/clip-vit-large-patch14`` ships it, CLIP-G as SD 3's ``text_encoder_2``. The tokenizer is
``openai/clip-vit-large-patch14``'s (MIT); CLIP-G shares its vocabulary and merges byte for byte. Vocabulary and
merges are stored gzip-compressed (see ``invokeai.backend.util.bundled_tokenizer``).
"""

import json
from functools import lru_cache
from pathlib import Path

from transformers import CLIPTextConfig, CLIPTokenizer

from invokeai.backend.model_manager.taxonomy import ClipVariantType
from invokeai.backend.util.bundled_tokenizer import load_gzipped_tokenizer_dir

_PACKAGE_DIR = Path(__file__).parent

_TEXT_CONFIG_FILES = {
    ClipVariantType.L: "clip_l_text_config.json",
    ClipVariantType.G: "clip_g_text_config.json",
}


@lru_cache(maxsize=1)
def load_bundled_clip_tokenizer() -> CLIPTokenizer:
    """Load the vendored CLIP tokenizer. Result is cached for the process."""
    tokenizer = load_gzipped_tokenizer_dir(_PACKAGE_DIR / "tokenizer")
    assert isinstance(tokenizer, CLIPTokenizer)
    return tokenizer


@lru_cache(maxsize=None)
def clip_text_config(variant: ClipVariantType) -> CLIPTextConfig:
    """The text tower architecture of a CLIP variant."""
    return CLIPTextConfig.from_dict(
        json.loads((_PACKAGE_DIR / _TEXT_CONFIG_FILES[variant]).read_text(encoding="utf-8"))
    )
