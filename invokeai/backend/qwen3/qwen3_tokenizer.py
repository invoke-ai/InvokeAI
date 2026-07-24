"""Bundled Qwen3 tokenizer for single-file / GGUF Qwen3 encoders.

Single-file (safetensors) and GGUF Qwen3 encoder checkpoints ship weights only —
no tokenizer files. Previously the tokenizer was pulled from HuggingFace
(``Qwen/Qwen3-4B``) on first use, which fails in offline / airgapped setups and
whenever the HF cache is not persisted. The Qwen3 BPE tokenizer is identical
across the 0.6B / 4B / 8B variants, so a single self-contained copy (Apache-2.0,
vendored from ``Qwen/Qwen3-4B``) serves every Qwen3 encoder, fully offline.
"""

import gzip
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase

_TOKENIZER_DIR = Path(__file__).parent / "tokenizer"

# tokenizer.json is ~11MB uncompressed, over the repo's 10MB LFS threshold, so it is
# vendored gzip-compressed and expanded into a temp dir at load time.
_TOKENIZER_JSON_GZ = _TOKENIZER_DIR / "tokenizer.json.gz"


@lru_cache(maxsize=1)
def load_bundled_qwen3_tokenizer() -> PreTrainedTokenizerBase:
    """Load the vendored Qwen3 fast tokenizer. Result is cached for the process."""
    # The fast tokenizer reads the vocab fully into memory at construction, so the
    # decompressed files only need to exist for the duration of from_pretrained().
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        with gzip.open(_TOKENIZER_JSON_GZ, "rb") as src, open(tmp_dir / "tokenizer.json", "wb") as dst:
            shutil.copyfileobj(src, dst)
        for name in ("tokenizer_config.json", "special_tokens_map.json"):
            shutil.copyfile(_TOKENIZER_DIR / name, tmp_dir / name)
        return AutoTokenizer.from_pretrained(tmp_dir, local_files_only=True)
