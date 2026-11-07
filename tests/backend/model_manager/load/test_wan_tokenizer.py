import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import sentencepiece as spm
from tokenizers import Tokenizer
from tokenizers.models import Unigram

from invokeai.backend.wan.tokenizer import load_wan_tokenizer


def _write_serialized_umt5_tokenizer(path: Path) -> None:
    backend = Tokenizer(
        Unigram(
            [("<pad>", 0.0), ("</s>", 0.0), ("<s>", 0.0), ("<unk>", 0.0), ("music", -1.0)],
            unk_id=3,
        )
    )
    backend.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "T5Tokenizer",
                "extra_ids": 0,
                "pad_token": "<pad>",
                "eos_token": "</s>",
                "bos_token": "<s>",
                "unk_token": "<unk>",
            }
        ),
        encoding="utf-8",
    )


def test_load_wan_tokenizer_preserves_serialized_unknown_token_id(tmp_path) -> None:
    """Unknown text must use UMT5's <unk> id, not its <s> id."""
    backend = Tokenizer(
        Unigram(
            [("<pad>", 0.0), ("</s>", 0.0), ("<s>", 0.0), ("<unk>", 0.0), ("music", -1.0)],
            unk_id=3,
        )
    )
    backend.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "T5Tokenizer",
                "extra_ids": 0,
                "pad_token": "<pad>",
                "eos_token": "</s>",
                "bos_token": "<s>",
                "unk_token": "<unk>",
            }
        ),
        encoding="utf-8",
    )

    tokenizer = load_wan_tokenizer(tmp_path)

    assert tokenizer.unk_token_id == 3
    assert tokenizer.encode("𝄞", add_special_tokens=False) == [3]


def test_load_wan_tokenizer_supplies_missing_umt5_special_token_config(tmp_path) -> None:
    backend = Tokenizer(
        Unigram(
            [("<pad>", 0.0), ("</s>", 0.0), ("<s>", 0.0), ("<unk>", 0.0), ("music", -1.0)],
            unk_id=3,
        )
    )
    backend.save(str(tmp_path / "tokenizer.json"))

    tokenizer = load_wan_tokenizer(tmp_path)

    assert tokenizer.pad_token_id == 0
    assert tokenizer.eos_token_id == 1
    assert tokenizer.bos_token_id == 2
    assert tokenizer.unk_token_id == 3
    encoded = tokenizer("music", padding="max_length", max_length=3)
    assert encoded["input_ids"] == [4, 1, 0]
    assert encoded["attention_mask"] == [1, 1, 0]


def test_load_wan_tokenizer_reconstructs_umt5_pipeline_without_tokenizer_json(tmp_path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("music melody video prompt\nhello world generation\na b\n", encoding="utf-8")
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(tokenizer_dir / "spiece"),
        model_type="unigram",
        vocab_size=300,
        hard_vocab_limit=False,
        byte_fallback=True,
        pad_id=0,
        pad_piece="<pad>",
        eos_id=1,
        eos_piece="</s>",
        bos_id=2,
        bos_piece="<s>",
        unk_id=3,
        unk_piece="<unk>",
        user_defined_symbols=["[eod]"],
        minloglevel=2,
    )
    sentencepiece = spm.SentencePieceProcessor(model_file=str(tokenizer_dir / "spiece.model"))

    tokenizer = load_wan_tokenizer(tokenizer_dir)

    for text in ("music", "  music", "[eod]", "a[eod]b"):
        expected = sentencepiece.encode(text, out_type=int)
        assert tokenizer.encode(text, add_special_tokens=False) == expected
        assert tokenizer.encode(text) == [*expected, sentencepiece.eos_id()]
    unknown_ids = tokenizer.encode("𝄞", add_special_tokens=False)
    assert unknown_ids[-1] == sentencepiece.unk_id() == 3
    assert 2 not in unknown_ids
    assert tokenizer.encode("𝄞") == [*unknown_ids, sentencepiece.eos_id()]
    assert tokenizer.unk_token_id == sentencepiece.unk_id() == 3


@pytest.mark.parametrize("nested", [True, False], ids=["nested", "flat"])
def test_wan_t5_encoder_loader_uses_serialized_tokenizer_pipeline(tmp_path, nested: bool) -> None:
    from invokeai.backend.model_manager.load.model_loaders.wan import WanT5EncoderLoader
    from invokeai.backend.model_manager.taxonomy import SubModelType

    model_path = tmp_path / "umt5"
    model_path.mkdir()
    if nested:
        text_encoder_dir = model_path / "text_encoder"
        text_encoder_dir.mkdir()
        (text_encoder_dir / "config.json").write_text(json.dumps({"model_type": "umt5"}), encoding="utf-8")
        tokenizer_dir = model_path / "tokenizer"
        tokenizer_dir.mkdir()
    else:
        (model_path / "config.json").write_text(json.dumps({"model_type": "umt5"}), encoding="utf-8")
        tokenizer_dir = model_path
    _write_serialized_umt5_tokenizer(tokenizer_dir)

    loader = object.__new__(WanT5EncoderLoader)
    tokenizer = loader._load_model(SimpleNamespace(path=str(model_path)), SubModelType.Tokenizer)

    assert tokenizer.unk_token_id == 3
    assert tokenizer.encode("𝄞", add_special_tokens=False) == [3]


def test_wan_diffusers_loader_uses_serialized_tokenizer_pipeline(tmp_path) -> None:
    from invokeai.backend.model_manager.load.model_loaders.wan import WanDiffusersModel
    from invokeai.backend.model_manager.taxonomy import SubModelType

    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    backend = Tokenizer(
        Unigram(
            [("<pad>", 0.0), ("</s>", 0.0), ("<s>", 0.0), ("<unk>", 0.0), ("music", -1.0)],
            unk_id=3,
        )
    )
    backend.save(str(tokenizer_dir / "tokenizer.json"))
    (tokenizer_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "T5Tokenizer",
                "extra_ids": 0,
                "pad_token": "<pad>",
                "eos_token": "</s>",
                "bos_token": "<s>",
                "unk_token": "<unk>",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "WanPipeline", "tokenizer": ["transformers", "T5TokenizerFast"]}),
        encoding="utf-8",
    )
    loader = object.__new__(WanDiffusersModel)

    def fail_if_model_index_class_is_used(*_args, **_kwargs):
        raise AssertionError("the Wan tokenizer must not be rebuilt through the model-index T5 class")

    loader.get_hf_load_class = fail_if_model_index_class_is_used
    tokenizer = loader._load_model(SimpleNamespace(path=str(tmp_path)), SubModelType.Tokenizer)

    assert tokenizer.unk_token_id == 3
    assert tokenizer.encode("𝄞", add_special_tokens=False) == [3]
