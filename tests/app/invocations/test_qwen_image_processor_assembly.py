"""How the Qwen-Image conditioning node assembles its Qwen2.5-VL processor.

Two things are pinned here: the tokenizer comes from the caller (the model cache) rather than
being re-derived per generation, and every layout ends up with the release's image preprocessor
rather than transformers' class defaults — those cap a reference image at 1,003,520 pixels against
the release's 12,845,056.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from invokeai.app.invocations.text_encoder.qwen_image_text_encoder import QwenImageTextEncoderInvocation
from invokeai.backend.qwen2_5_vl.qwen2_5_vl_assets import load_bundled_qwen2_5_vl_preprocessor_config_dict

RELEASE_MAX_PIXELS = 12845056
CLASS_DEFAULT_MAX_PIXELS = 1003520


@pytest.fixture
def tokenizer(tmp_path_factory):
    """A real (tiny) fast tokenizer: `Qwen2_5_VLProcessor` type-checks what it is handed and reads
    special-token ids off it, so a double would only test the double.
    """
    from transformers import PreTrainedTokenizerFast

    spec = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    spec.write_text(
        json.dumps(
            {
                "version": "1.0",
                "added_tokens": [],
                "normalizer": None,
                "pre_tokenizer": None,
                "post_processor": None,
                "decoder": None,
                "model": {
                    "type": "WordLevel",
                    "vocab": {"<|image_pad|>": 0, "<|video_pad|>": 1, "a": 2},
                    "unk_token": "a",
                },
            }
        ),
        encoding="utf-8",
    )
    return PreTrainedTokenizerFast(tokenizer_file=str(spec))


def test_the_callers_tokenizer_is_used_rather_than_one_built_here(tmp_path: Path, tokenizer) -> None:
    """The cache bypass this replaced: the node used to re-derive the tokenizer from `model_root`,
    parsing a ~7MB vocabulary per generation into a copy the model cache knew nothing about.
    """
    checkpoint = tmp_path / "qwen_2.5_vl_7b_fp8_scaled.safetensors"
    checkpoint.write_bytes(b"not read here")

    assert QwenImageTextEncoderInvocation._build_processor(tokenizer, checkpoint).tokenizer is tokenizer


def test_a_single_file_checkpoint_uses_the_bundled_preprocessor(tmp_path: Path, tokenizer) -> None:
    """model_root is the safetensors file itself: no folder beside it to read a config from."""
    checkpoint = tmp_path / "encoder.safetensors"
    checkpoint.write_bytes(b"")

    processor = QwenImageTextEncoderInvocation._build_processor(tokenizer, checkpoint)

    assert processor.image_processor.size.longest_edge == RELEASE_MAX_PIXELS


def test_a_folder_install_without_a_preprocessor_config_still_gets_the_release_budget(
    tmp_path: Path, tokenizer
) -> None:
    """The last rung used to be a bare constructor, which silently downscaled reference images.

    Reachable for anyone who installed only text_encoder + tokenizer, with no processor folder.
    """
    model_root = tmp_path / "qwen-vl"
    (model_root / "tokenizer").mkdir(parents=True)

    processor = QwenImageTextEncoderInvocation._build_processor(tokenizer, model_root)

    assert processor.image_processor.size.longest_edge == RELEASE_MAX_PIXELS
    assert processor.image_processor.size.longest_edge != CLASS_DEFAULT_MAX_PIXELS


def test_a_folder_install_prefers_its_own_preprocessor_config(tmp_path: Path, tokenizer) -> None:
    """A complete install keeps deciding for itself; the bundle is a fallback, not an override."""
    model_root = tmp_path / "qwen-vl"
    (model_root / "tokenizer").mkdir(parents=True)
    installed = dict(load_bundled_qwen2_5_vl_preprocessor_config_dict())
    installed["max_pixels"] = 4096 * 4096
    (model_root / "processor").mkdir()
    (model_root / "processor" / "preprocessor_config.json").write_text(json.dumps(installed), encoding="utf-8")

    processor = QwenImageTextEncoderInvocation._build_processor(tokenizer, model_root)

    assert processor.image_processor.size.longest_edge == 4096 * 4096


@pytest.mark.parametrize("search_dir", ["processor", "tokenizer", "", "image_processor"])
def test_an_installed_preprocessor_config_is_found_in_every_searched_location(
    tmp_path: Path, tokenizer, search_dir: str
) -> None:
    """Pins the search order's membership: a dropped entry would silently fall back to the bundle
    for installs that do carry their own config.
    """
    model_root = tmp_path / "qwen-vl"
    (model_root / "tokenizer").mkdir(parents=True)
    installed = dict(load_bundled_qwen2_5_vl_preprocessor_config_dict())
    installed["max_pixels"] = 2048 * 2048
    target = model_root / search_dir if search_dir else model_root
    target.mkdir(parents=True, exist_ok=True)
    (target / "preprocessor_config.json").write_text(json.dumps(installed), encoding="utf-8")

    processor = QwenImageTextEncoderInvocation._build_processor(tokenizer, model_root)

    assert processor.image_processor.size.longest_edge == 2048 * 2048


def test_a_tokenizer_that_cannot_encode_is_refused_whatever_produced_it(tmp_path: Path) -> None:
    """Every tokenizer source converges on `_build_processor`, which is why the probe sits there.

    A `tokenizer/` that lost its vocabulary does not make `from_pretrained` raise; it yields a
    one-token vocabulary that encodes every prompt to an empty sequence. Guarding one loader would
    have left the diffusers main-model route — the one most installs use — silently unprotected.
    """

    class _Degenerate:
        def __call__(self, *_a, **_k):
            return {"input_ids": []}

    checkpoint = tmp_path / "encoder.safetensors"
    checkpoint.write_bytes(b"")

    with pytest.raises(RuntimeError, match="cannot encode anything"):
        QwenImageTextEncoderInvocation._build_processor(_Degenerate(), checkpoint)


def test_the_node_takes_its_tokenizer_from_the_model_cache(tmp_path: Path, tokenizer, monkeypatch) -> None:
    """The cache bypass this replaced: re-deriving the tokenizer from disk per generation cost a
    ~7MB vocabulary parse and produced a second copy the model cache knew nothing about.

    Asserted on `context.models.load` rather than on timing, so reverting that one line fails here
    instead of quietly restoring the cost.
    """
    import torch

    from invokeai.app.invocations.model import ModelIdentifierField, QwenVLEncoderField
    from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType

    encoder = ModelIdentifierField(
        key="enc", hash="h", name="qwen-vl", base=BaseModelType.QwenImage, type=ModelType.QwenVLEncoder
    )
    invocation = QwenImageTextEncoderInvocation(
        prompt="a cat",
        qwen_vl_encoder=QwenVLEncoderField(tokenizer=encoder, text_encoder=encoder),
        quantization="none",
    )

    model_root = tmp_path / "qwen-vl"
    (model_root / "tokenizer").mkdir(parents=True)
    context = MagicMock()
    context.models.get_absolute_path.return_value = model_root
    context.models.load.return_value.model = tokenizer

    seen: dict[str, object] = {}
    monkeypatch.setattr(
        QwenImageTextEncoderInvocation,
        "_build_processor",
        staticmethod(lambda tok, root: seen.update(tokenizer=tok, root=root) or MagicMock()),
    )
    monkeypatch.setattr(
        QwenImageTextEncoderInvocation,
        "_load_cached_encoder",
        lambda _self, _ctx: (torch.nn.Identity(), torch.device("cpu"), lambda: None),
    )
    monkeypatch.setattr(
        QwenImageTextEncoderInvocation,
        "_run_encoder",
        staticmethod(lambda *_a, **_k: (torch.zeros(1, 2, 4), torch.ones(1, 2, dtype=torch.long))),
    )

    invocation._encode(context, images=[])

    context.models.load.assert_called_once_with(invocation.qwen_vl_encoder.tokenizer)
    assert seen["tokenizer"] is tokenizer, "the node built its own tokenizer instead of using the cached one"
