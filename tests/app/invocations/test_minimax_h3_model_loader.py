"""MiniMax H3 model loader input validation.

The Model field needs the diffusers-folder install — a single-file checkpoint main
(e.g. the pruned int8 transformer repack) carries none of the folder submodels this
node fans out. The picker filters on ui_model_format, but hand-authored workflows and
clients that ignore the hint can still send one; the node must fail fast with an
actionable message instead of an opaque loader stack trace minutes later.
"""

from unittest.mock import MagicMock

import pytest

from invokeai.app.invocations.minimax_h3.minimax_h3_model_loader import MiniMaxH3ModelLoaderInvocation
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.backend.model_manager.taxonomy import BaseModelType, MiniMaxH3VariantType, ModelFormat, ModelType


def _identifier(key: str = "h3-main") -> ModelIdentifierField:
    return ModelIdentifierField(
        key=key, hash="test-hash", name="MiniMax H3", base=BaseModelType.MiniMaxH3, type=ModelType.Main
    )


def _context(config: MagicMock) -> MagicMock:
    context = MagicMock()
    context.models.exists.return_value = True
    context.models.get_config.return_value = config
    return context


def _config(base=BaseModelType.MiniMaxH3, type_=ModelType.Main, format_=ModelFormat.Diffusers) -> MagicMock:
    config = MagicMock()
    config.base = base
    config.type = type_
    config.format = format_
    config.name = "MiniMax H3 FL2VA Transformer (int8, pruned)"
    return config


def test_rejects_single_file_checkpoint_as_main_model():
    node = MiniMaxH3ModelLoaderInvocation(id="loader", model=_identifier())
    with pytest.raises(ValueError, match="single-file checkpoint.*Transformer \\(single file\\)"):
        node.invoke(_context(_config(format_=ModelFormat.Checkpoint)))


def test_rejects_wrong_base_or_type():
    node = MiniMaxH3ModelLoaderInvocation(id="loader", model=_identifier())
    with pytest.raises(ValueError, match="not a MiniMax H3 main model"):
        node.invoke(_context(_config(base=BaseModelType.Wan)))
    with pytest.raises(ValueError, match="not a MiniMax H3 main model"):
        node.invoke(_context(_config(type_=ModelType.LoRA)))


def test_accepts_diffusers_folder_main():
    node = MiniMaxH3ModelLoaderInvocation(id="loader", model=_identifier())
    output = node.invoke(_context(_config()))
    assert output.transformer.transformer.key == "h3-main"
    assert output.vae.vae.key == "h3-main"


def _context_with_override(main_config: MagicMock, override_config: MagicMock) -> MagicMock:
    context = MagicMock()
    context.models.exists.return_value = True
    context.models.get_config.side_effect = lambda key: override_config if key == "h3-ckpt" else main_config
    return context


def test_stamps_ref2va_variant_from_transformer_override():
    # The denoise node keys its task/conditioning guard off this stamp - references on FL2VA
    # weights (or Ref2VA weights without references) must fail fast there.
    ref2va = _config(format_=ModelFormat.Checkpoint)
    ref2va.variant = MiniMaxH3VariantType.REF2VA
    ref2va.name = "MiniMax H3 Ref2VA Transformer (int8, pruned)"
    node = MiniMaxH3ModelLoaderInvocation(id="loader", model=_identifier(), transformer_model=_identifier("h3-ckpt"))
    output = node.invoke(_context_with_override(_config(), ref2va))
    assert output.transformer.transformer.key == "h3-ckpt"
    assert output.transformer.variant == "ref2va"


def test_stamps_fl2va_variant_from_transformer_override():
    fl2va = _config(format_=ModelFormat.Checkpoint)
    fl2va.variant = MiniMaxH3VariantType.FL2VA
    node = MiniMaxH3ModelLoaderInvocation(id="loader", model=_identifier(), transformer_model=_identifier("h3-ckpt"))
    output = node.invoke(_context_with_override(_config(), fl2va))
    assert output.transformer.transformer.key == "h3-ckpt"
    assert output.transformer.variant == "fl2va"


def test_stamps_variant_from_diffusers_main_without_override():
    main = _config()
    main.variant = MiniMaxH3VariantType.REF2VA
    node = MiniMaxH3ModelLoaderInvocation(id="loader", model=_identifier())
    output = node.invoke(_context(main))
    assert output.transformer.variant == "ref2va"


def test_variant_stamp_is_none_for_untyped_configs():
    # A MagicMock config's `variant` attribute is not a MiniMaxH3VariantType, standing in for
    # any config without one - the stamp must degrade to None, not an arbitrary object.
    node = MiniMaxH3ModelLoaderInvocation(id="loader", model=_identifier())
    output = node.invoke(_context(_config()))
    assert output.transformer.variant is None
