"""Bundled T5-XXL tokenizer for Anima.

Anima tokenizes the prompt with the T5-XXL tokenizer to produce token IDs that
index the LLM Adapter's learned embedding table. Only the tokenizer is needed —
never the 9GB T5-XXL weights — so the tokenizer is vendored in the package as a
self-contained fast tokenizer (tokenizer.json), avoiding both the large download
and the sentencepiece runtime path.
"""

from functools import lru_cache
from pathlib import Path

from transformers import T5TokenizerFast

# Size of the LLM Adapter's token embedding table (T5 v1.1 vocab incl. 100 sentinel
# extra_id tokens). Token IDs must stay within this range.
ANIMA_T5_VOCAB_SIZE = 32128

_TOKENIZER_DIR = Path(__file__).parent / "tokenizer"


@lru_cache(maxsize=1)
def load_bundled_t5_tokenizer() -> T5TokenizerFast:
    """Load the vendored T5-XXL fast tokenizer. Result is cached for the process."""
    return T5TokenizerFast.from_pretrained(_TOKENIZER_DIR)
