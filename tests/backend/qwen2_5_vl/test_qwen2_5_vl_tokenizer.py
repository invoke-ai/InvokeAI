"""Tests for the vendored Qwen2.5-VL tokenizer and the probe that guards local installs.

Single-file Qwen2.5-VL encoders ship weights only, so the tokenizer is vendored and no longer
fetched. The probe stays because the same silent failure is reachable from a folder-layout install
whose ``tokenizer/`` lost its vocabulary — vendoring cannot repair somebody else's install.
"""

import json
from pathlib import Path

from invokeai.backend.qwen2_5_vl.qwen2_5_vl_assets import (
    load_bundled_qwen2_5_vl_tokenizer,
    tokenizer_can_encode,
)

# Qwen2.5-VL's vocabulary. Token ids stay below this.
QWEN2_5_VL_VOCAB_SIZE = 152064


def test_bundled_tokenizer_known_ids() -> None:
    """Pinned ids, not just "non-empty": a wrong-but-working vocabulary encodes fine and conditions
    off-distribution. These are what ``Qwen/Qwen2.5-VL-7B-Instruct`` produces for this prompt.
    """
    tokenizer = load_bundled_qwen2_5_vl_tokenizer()

    assert tokenizer("A cinematic photo of a cat").input_ids == [32, 64665, 6548, 315, 264, 8251]


def test_bundled_tokenizer_roundtrip() -> None:
    tokenizer = load_bundled_qwen2_5_vl_tokenizer()
    prompt = "A cinematic photo of a cat"

    assert tokenizer.decode(tokenizer(prompt).input_ids) == prompt


def test_bundled_tokenizer_encodes_a_rare_prompt_within_the_vocabulary() -> None:
    ids = load_bundled_qwen2_5_vl_tokenizer()("a rare one: zxqwv 12345").input_ids

    assert ids
    assert all(0 <= i < QWEN2_5_VL_VOCAB_SIZE for i in ids)


def test_bundled_tokenizer_carries_the_vision_special_tokens() -> None:
    """The conditioning splices one vision placeholder per reference image. A tokenizer that
    shattered those markers into ordinary BPE pieces would still encode, just wrongly.
    """
    tokenizer = load_bundled_qwen2_5_vl_tokenizer()

    for marker in ("<|vision_start|>", "<|vision_end|>", "<|image_pad|>"):
        assert len(tokenizer(marker, add_special_tokens=False).input_ids) == 1


def test_bundled_tokenizer_is_cached() -> None:
    """Rebuilt per call this would parse a ~7MB vocabulary on every generation."""
    assert load_bundled_qwen2_5_vl_tokenizer() is load_bundled_qwen2_5_vl_tokenizer()


def test_the_probe_rejects_a_one_token_vocabulary() -> None:
    """The exact shape a vocabulary-less directory produces: it encodes, but to nothing."""

    class _Degenerate:
        def __call__(self, *_a, **_k):
            return {"input_ids": []}

    assert not tokenizer_can_encode(_Degenerate())


def test_the_probe_accepts_the_bundled_tokenizer() -> None:
    assert tokenizer_can_encode(load_bundled_qwen2_5_vl_tokenizer())


def test_the_probe_treats_a_raising_tokenizer_as_unusable() -> None:
    class _Broken:
        def __call__(self, *_a, **_k):
            raise TypeError("slow tokenizer path with no vocab files")

    assert not tokenizer_can_encode(_Broken())


def test_the_probe_rejects_what_transformers_returns_for_a_vocabulary_less_directory(tmp_path: Path) -> None:
    """Not a hypothetical: an interrupted model install leaves exactly this on disk, and
    ``from_pretrained`` loads it without complaint.
    """
    from transformers import AutoTokenizer

    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "Qwen2Tokenizer"}), encoding="utf-8")

    degenerate = AutoTokenizer.from_pretrained(str(tmp_path), local_files_only=True)

    assert degenerate("a photo of a cat").input_ids == []
    assert not tokenizer_can_encode(degenerate)


def test_the_diffusers_main_loader_serves_a_usable_tokenizer(tmp_path: Path) -> None:
    """A path this change made live.

    The conditioning node used to build the tokenizer itself, so for a Diffusers Qwen-Image main
    model the ``Tokenizer`` submodel had never actually been loaded through the cache. Now it is,
    via ``get_hf_load_class`` with ``dtype``/``variant`` kwargs that a tokenizer does not expect —
    harmless on the pinned transformers, and this is what says so out loud.
    """
    import gzip
    import shutil

    from invokeai.backend.model_manager.configs.main import Main_Diffusers_QwenImage_Config
    from invokeai.backend.model_manager.load.model_loaders.qwen_image import QwenImageDiffusersModel
    from invokeai.backend.model_manager.taxonomy import SubModelType
    from invokeai.backend.qwen2_5_vl import qwen2_5_vl_assets

    # Stage the vendored tokenizer as a diffusers folder install would carry it.
    vendored = Path(qwen2_5_vl_assets.__file__).parent / "tokenizer"
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    with gzip.open(vendored / "tokenizer.json.gz", "rb") as src, open(tokenizer_dir / "tokenizer.json", "wb") as dst:
        shutil.copyfileobj(src, dst)
    shutil.copyfile(vendored / "tokenizer_config.json", tokenizer_dir / "tokenizer_config.json")

    # What `get_hf_load_class` reads to decide which class serves this submodel, as a real
    # diffusers Qwen-Image install carries it.
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "QwenImagePipeline", "tokenizer": ["transformers", "Qwen2Tokenizer"]}),
        encoding="utf-8",
    )

    loader = object.__new__(QwenImageDiffusersModel)
    config = Main_Diffusers_QwenImage_Config.model_construct(path=str(tmp_path), name="qwen-image", repo_variant=None)

    tokenizer = loader._load_model(config, SubModelType.Tokenizer)

    assert tokenizer.is_fast
    assert tokenizer("A cinematic photo of a cat").input_ids == [32, 64665, 6548, 315, 264, 8251]
    assert tokenizer_can_encode(tokenizer)
