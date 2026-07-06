"""Bundled Qwen3 tokenizer for single-file / GGUF Qwen3 encoders.

Single-file (safetensors) and GGUF Qwen3 encoder checkpoints ship weights only —
no tokenizer files. Previously the tokenizer was pulled from HuggingFace
(``Qwen/Qwen3-4B``) on first use, which fails in offline / airgapped setups and
whenever the HF cache is not persisted. The Qwen3 BPE tokenizer is identical
across the 0.6B / 4B / 8B variants, so a single self-contained copy (Apache-2.0,
vendored from ``Qwen/Qwen3-4B``) serves every Qwen3 encoder, fully offline.
"""

from functools import lru_cache
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase

_TOKENIZER_DIR = Path(__file__).parent / "tokenizer"


@lru_cache(maxsize=1)
def load_bundled_qwen3_tokenizer() -> PreTrainedTokenizerBase:
    """Load the vendored Qwen3 fast tokenizer. Result is cached for the process."""
    return AutoTokenizer.from_pretrained(_TOKENIZER_DIR, local_files_only=True)
