"""Vendored tokenizer, architecture config and image preprocessor config for Qwen2.5-VL.

Single-file Qwen2.5-VL encoder checkpoints (e.g. ComfyUI's fp8_scaled build) carry weights only,
so all three of these used to be fetched from ``Qwen/Qwen2.5-VL-7B-Instruct`` at load time. Each
had its own reason to stop:

- The config's constants decide the module tree the weights are folded into, and a re-upload
  changing one of them would change conditioning against unchanged weights, silently.
- The preprocessor config decides the reference-image pixel budget. The fallback when the hub was
  unreachable was a bare ``Qwen2VLImageProcessor()``, whose class defaults cap an image at
  1,003,520 pixels against the release's 12,845,056 — a ~3.6x downscale per side, unlogged.
- The tokenizer is the one whose absence was worst. A hub cache holding the repo's ``config.json``
  but none of its tokenizer files does not raise: ``AutoTokenizer`` returns a one-token vocabulary
  that encodes every prompt to an empty sequence, so the image is generated from no conditioning
  at all. InvokeAI itself used to create that cache state by fetching the config first.

Everything here is vendored verbatim from the Apache-2.0 release, so single-file Qwen-Image now
loads with no network access at all.

``tokenizer_can_encode`` remains because the same silent failure is reachable from a *local*
directory: a folder-layout install whose ``tokenizer/`` lost its vocabulary loads just as
successfully, and just as uselessly. Vendoring cannot fix somebody else's incomplete install, so
the loader probes what it reads from disk.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizerBase

from invokeai.backend.util.bundled_tokenizer import load_gzipped_tokenizer_dir

_ASSET_DIR = Path(__file__).parent
_TOKENIZER_DIR = _ASSET_DIR / "tokenizer"
_CONFIG_PATH = _ASSET_DIR / "qwen2_5_vl_7b_instruct_config.json"
_PREPROCESSOR_CONFIG_PATH = _ASSET_DIR / "qwen2_5_vl_7b_instruct_preprocessor_config.json"


@lru_cache(maxsize=1)
def load_bundled_qwen2_5_vl_tokenizer() -> PreTrainedTokenizerBase:
    """Load the vendored Qwen2.5-VL fast tokenizer. Result is cached for the process."""
    return load_gzipped_tokenizer_dir(_TOKENIZER_DIR)


def load_bundled_qwen2_5_vl_config_dict() -> dict[str, Any]:
    """The vendored ``Qwen/Qwen2.5-VL-7B-Instruct`` ``config.json``, as published.

    Returns a fresh dict per call: the loader sets ``torch_dtype`` on the config it builds from it.
    """
    config_dict: dict[str, Any] = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    return config_dict


def load_bundled_qwen2_5_vl_preprocessor_config_dict() -> dict[str, Any]:
    """The vendored ``preprocessor_config.json``, as published.

    Handed to ``Qwen2VLImageProcessor.from_dict`` rather than a directory: it is the only asset
    the image processor needs, so staging a directory for it would buy nothing.
    """
    config_dict: dict[str, Any] = json.loads(_PREPROCESSOR_CONFIG_PATH.read_text(encoding="utf-8"))
    return config_dict


def tokenizer_can_encode(tokenizer: Any) -> bool:
    """Whether a tokenizer that loaded successfully is actually usable.

    Not paranoia, and not only about the hub: a directory holding ``tokenizer_config.json`` but no
    vocabulary — an interrupted model install, or a wheel with a bad data glob — does not make
    ``from_pretrained`` raise. It returns a ``Qwen2Tokenizer`` with a one-token vocabulary and no
    chat template, which encodes every prompt to an empty sequence. The generation then runs on no
    conditioning at all, and nothing in the log says so. Probing the round trip tells the two apart.
    """
    try:
        return bool(tokenizer("probe", add_special_tokens=False)["input_ids"])
    except Exception:
        return False
