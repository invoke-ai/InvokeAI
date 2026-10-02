"""Vendored tokenizer and architecture configs for the single-file Qwen3-VL encoders.

Single-file safetensors and GGUF Qwen3-VL encoders ship weights only, so the tokenizer and the
architecture config used to be fetched from ``Qwen/Qwen3-VL-{4B,8B}-Instruct`` at load time. Three
things make that the wrong source of truth:

- A partially populated HF cache — the repo's ``config.json`` present, its tokenizer files not —
  does not raise. ``AutoTokenizer`` returns a ``Qwen2Tokenizer`` with a one-token vocabulary that
  encodes every prompt to an empty sequence, and the generation then runs on no conditioning at
  all with nothing in the log to say so. Fetching the config first is exactly what left the cache
  in that state, so the failure was reachable on an ordinary first run.
- Architecture constants can change under a re-upload. A wrong ``rope_theta`` (1e6 rather than
  Qwen3-VL's 5e6) costs relative L2 0.1008 against otherwise identical weights — as much as a full
  Q4 quantization, and just as silent.
- Offline, air-gapped and proxied installs have no route to the hub at all.

The tokenizer is byte-identical across the 4B and 8B releases — verified by md5 against both
snapshots when it was vendored, which is why one copy serves both, in both container formats. That
identity cannot be re-checked here without the hub, so what the tests pin instead is the behaviour
that depends on it: the exact token ids and the single-id vision markers. Configs differ per
variant and are vendored per variant.

Everything here is vendored verbatim from the Apache-2.0 ``Qwen/Qwen3-VL-{4B,8B}-Instruct``
releases. The configs are deliberately *not* normalized on the way in: whether ``rope_parameters``
has to be mirrored onto ``rope_scaling`` depends on the installed transformers version, not on the
file, so that stays in the load path (`normalize_qwen3vl_rope_config`) where it can keep tracking
transformers. Freezing a normalized config here would bake in today's transformers instead.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizerBase

from invokeai.backend.model_manager.taxonomy import Qwen3VLVariantType
from invokeai.backend.util.bundled_tokenizer import load_gzipped_tokenizer_dir

_ASSET_DIR = Path(__file__).parent
_TOKENIZER_DIR = _ASSET_DIR / "tokenizer"
_CONFIG_DIR = _ASSET_DIR / "config"

_CONFIG_BY_VARIANT: dict[Qwen3VLVariantType, str] = {
    Qwen3VLVariantType.Qwen3VL_4B: "qwen3_vl_4b_instruct.json",
    Qwen3VLVariantType.Qwen3VL_8B: "qwen3_vl_8b_instruct.json",
}


@lru_cache(maxsize=1)
def load_bundled_qwen3_vl_tokenizer() -> PreTrainedTokenizerBase:
    """Load the vendored Qwen3-VL fast tokenizer. Result is cached for the process.

    ``extra_special_tokens={}`` matches what the loaders passed when they fetched this tokenizer
    from the hub, so the returned object is configured exactly as before.
    """
    return load_gzipped_tokenizer_dir(_TOKENIZER_DIR, extra_special_tokens={})


def load_bundled_qwen3_vl_config_dict(variant: Qwen3VLVariantType) -> dict[str, Any]:
    """The vendored ``config.json`` for a Qwen3-VL variant, as published.

    Returns a fresh dict per call so that a caller mutating it — as `minimax_h3` does with its own
    bundled config, to truncate the layer count — cannot corrupt the copy the next load sees.
    """
    filename = _CONFIG_BY_VARIANT.get(variant)
    if filename is None:
        raise NotImplementedError(
            f"No vendored Qwen3-VL config for variant '{variant.value}'. Add one to "
            f"{_CONFIG_DIR} and register it in _CONFIG_BY_VARIANT: a single-file encoder of this "
            "variant carries no config of its own, so there is nothing to fall back to."
        )
    config_dict: dict[str, Any] = json.loads((_CONFIG_DIR / filename).read_text(encoding="utf-8"))
    return config_dict
