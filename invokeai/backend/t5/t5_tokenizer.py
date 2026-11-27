"""Bundled T5-XXL tokenizer.

Several features need to tokenize prompts with the T5 v1.1 XXL tokenizer without ever loading the
~9GB T5-XXL weights:
- Anima tokenizes prompts to index the LLM Adapter's learned embedding table.
- The GGUF T5 encoder loader needs a tokenizer since single-file GGUFs don't ship one that
  transformers can read.

The tokenizer (Apache-2.0, same as google/t5-v1_1-xxl) is vendored here as a self-contained fast
tokenizer (tokenizer.json), avoiding both the large download and the sentencepiece runtime path.
"""

from functools import lru_cache
from pathlib import Path

from transformers import T5TokenizerFast

# T5 v1.1 vocabulary size, including the 100 sentinel extra_id tokens. Token IDs stay within this range.
T5_VOCAB_SIZE = 32128

_TOKENIZER_DIR = Path(__file__).parent / "tokenizer"


@lru_cache(maxsize=1)
def load_bundled_t5_tokenizer() -> T5TokenizerFast:
    """Load the vendored T5-XXL fast tokenizer. Result is cached for the process."""
    return T5TokenizerFast.from_pretrained(_TOKENIZER_DIR)
