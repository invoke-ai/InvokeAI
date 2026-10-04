"""Load a tokenizer from a vendored directory whose large files are stored gzip-compressed.

Two things make this indirection necessary rather than a wrapper around ``from_pretrained``:

- ``.gitattributes`` sets ``* text=auto``, so a raw ``tokenizer.json`` checked into the repository
  is line-ending normalized per platform and no longer matches the bytes upstream published. A
  gzip member is binary and survives that untouched, which is the whole point of vendoring: the
  vocabulary must be exactly the one the weights were trained against.
- ``tokenizer.json`` is the bulk of every vendored tokenizer (7-11MB raw, and the repository keeps
  files at or above 10MB in LFS). Compressed they are ~2MB each.

``from_pretrained`` only reads directories, so the compressed members are expanded into a
temporary directory for the duration of the call. The fast tokenizer parses the vocabulary fully
into memory during construction, so nothing reads those files afterwards.
"""

import gzip
import shutil
import tempfile
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase

GZIP_SUFFIX = ".gz"

# A tokenizer is only usable if one of these is staged; see `load_gzipped_tokenizer_dir`.
VOCABULARY_MEMBERS = frozenset({"tokenizer.json", "vocab.json", "spiece.model"})


def load_gzipped_tokenizer_dir(tokenizer_dir: Path, **kwargs: object) -> PreTrainedTokenizerBase:
    """Load the vendored tokenizer in ``tokenizer_dir``, expanding any ``*.gz`` members first.

    Files are staged rather than loaded in place, so the vendored directory is never written to.

    The vocabulary is checked for explicitly rather than left to ``from_pretrained``. A directory
    holding ``tokenizer_config.json`` but no vocabulary is not an error there: it yields a
    tokenizer with a one-token vocabulary that encodes every prompt to an empty sequence, so the
    generation runs on no conditioning with nothing in the log. A wheel whose package-data globs
    missed ``*.json.gz`` produces exactly that directory, and this is where that has to be caught.
    """
    members = sorted(p for p in tokenizer_dir.iterdir() if p.is_file()) if tokenizer_dir.is_dir() else []
    staged_names = {p.stem if p.suffix == GZIP_SUFFIX else p.name for p in members}
    if not VOCABULARY_MEMBERS & staged_names:
        raise FileNotFoundError(
            f"The vendored tokenizer in {tokenizer_dir} is missing its vocabulary "
            f"(expected one of {sorted(VOCABULARY_MEMBERS)}, found {sorted(staged_names) or 'nothing'}). "
            "This InvokeAI installation is incomplete -- reinstall the package."
        )
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp)
        for member in members:
            if member.suffix == GZIP_SUFFIX:
                with gzip.open(member, "rb") as src, open(staged / member.stem, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            else:
                shutil.copyfile(member, staged / member.name)
        return AutoTokenizer.from_pretrained(staged, local_files_only=True, **kwargs)
