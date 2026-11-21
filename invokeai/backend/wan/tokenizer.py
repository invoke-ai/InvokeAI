import json
from pathlib import Path
from typing import Any

_UMT5_SPECIAL_TOKENS = (
    ("pad_token", "<pad>"),
    ("eos_token", "</s>"),
    ("bos_token", "<s>"),
    ("unk_token", "<unk>"),
)


def _apply_umt5_special_token_defaults(tokenizer: Any) -> Any:
    """Fill UMT5's fixed special tokens without overriding folder metadata."""
    for token_name, token_value in _UMT5_SPECIAL_TOKENS:
        if getattr(tokenizer, token_name) is None:
            setattr(tokenizer, token_name, token_value)
    return tokenizer


def _install_umt5_eos_post_processor(tokenizer: Any) -> None:
    """Install the T5 EOS template when a serialized backend does not provide one."""
    from tokenizers.processors import TemplateProcessing

    tokenizer.backend_tokenizer.post_processor = TemplateProcessing(
        single=["$A", tokenizer.eos_token],
        pair=["$A", tokenizer.eos_token, "$B", tokenizer.eos_token],
        special_tokens=[(tokenizer.eos_token, tokenizer.eos_token_id)],
    )


def _load_sentencepiece_tokenizer(tokenizer_path: Path) -> Any:
    """Convert a standalone UMT5 SentencePiece model without rebuilding it as T5."""
    from sentencepiece import sentencepiece_model_pb2
    from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
    from tokenizers.models import Unigram
    from transformers import PreTrainedTokenizerFast

    model_proto = sentencepiece_model_pb2.ModelProto()
    model_proto.ParseFromString((tokenizer_path / "spiece.model").read_bytes())
    trainer_spec = model_proto.trainer_spec
    if trainer_spec.model_type != sentencepiece_model_pb2.TrainerSpec.UNIGRAM:
        raise ValueError(f"Wan UMT5 requires a Unigram SentencePiece model: {tokenizer_path / 'spiece.model'}")

    backend = Tokenizer(
        Unigram(
            [(piece.piece, piece.score) for piece in model_proto.pieces],
            unk_id=trainer_spec.unk_id,
            # The published UMT5 tokenizer.json deliberately disables the
            # SentencePiece model's byte fallback. Wan was trained against that
            # behavior, where unsupported text maps to <unk> rather than bytes.
            byte_fallback=False,
        )
    )
    # Only UMT5's named special tokens are AddedTokens in the published
    # tokenizer.json. SentencePiece USER_DEFINED pieces such as ``[eod]`` must
    # remain ordinary Unigram pieces so metaspace boundaries are preserved.
    named_special_tokens = {token_value for _, token_value in _UMT5_SPECIAL_TOKENS}
    added_tokens = [
        AddedToken(piece.piece, normalized=False, special=True)
        for piece in model_proto.pieces
        if piece.piece in named_special_tokens
    ]
    if added_tokens:
        backend.add_tokens(added_tokens)

    tokenizer_normalizers = []
    if model_proto.normalizer_spec.precompiled_charsmap:
        tokenizer_normalizers.append(normalizers.Precompiled(model_proto.normalizer_spec.precompiled_charsmap))
    tokenizer_normalizers.append(normalizers.Replace(Regex(" {2,}"), " "))
    backend.normalizer = normalizers.Sequence(tokenizer_normalizers)
    backend.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
    backend.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")

    if trainer_spec.eos_id >= 0:
        eos_token = trainer_spec.eos_piece or "</s>"
        backend.post_processor = processors.TemplateProcessing(
            single=["$A", eos_token],
            pair=["$A", eos_token, "$B", eos_token],
            special_tokens=[(eos_token, trainer_spec.eos_id)],
        )

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        str(tokenizer_path), tokenizer_object=backend, local_files_only=True
    )
    return _apply_umt5_special_token_defaults(tokenizer)


def load_wan_tokenizer(tokenizer_path: Path) -> Any:
    """Load a tokenizer for a standalone Wan UMT5 encoder."""
    if (tokenizer_path / "tokenizer.json").is_file():
        from transformers import PreTrainedTokenizerFast

        with (tokenizer_path / "tokenizer.json").open(encoding="utf-8") as tokenizer_file:
            has_serialized_post_processor = json.load(tokenizer_file).get("post_processor") is not None

        # Transformers 5's T5Tokenizer rebuilds a Unigram pipeline with unk_id=2.
        # UMT5 uses id 2 for <s> and serializes the correct unk_id=3 in tokenizer.json,
        # so load that pipeline directly while still applying the folder's token config.
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path), local_files_only=True)
        tokenizer = _apply_umt5_special_token_defaults(tokenizer)
        if not has_serialized_post_processor:
            _install_umt5_eos_post_processor(tokenizer)
        return tokenizer

    if (tokenizer_path / "spiece.model").is_file():
        return _load_sentencepiece_tokenizer(tokenizer_path)

    from transformers import AutoTokenizer

    # Keep support for non-standard tokenizer layouts handled by Transformers.
    return AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
