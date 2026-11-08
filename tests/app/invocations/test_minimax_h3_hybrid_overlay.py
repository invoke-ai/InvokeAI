"""The hybrid AdaLN overlay node: input validation and the transformer field it emits.

The node only records the overlay on the transformer field (the denoise node applies it), so what
it must get right is refusing pairs that cannot work - and stamping the task the hybrid serves.
"""

from unittest.mock import MagicMock

import pytest

from invokeai.app.invocations.minimax_h3.minimax_h3_hybrid_overlay import MiniMaxH3HybridOverlayInvocation
from invokeai.app.invocations.model import LoRAField, MiniMaxH3TransformerField, ModelIdentifierField
from invokeai.backend.model_manager.configs.lora import LoRA_LyCORIS_MiniMaxH3_Config
from invokeai.backend.model_manager.configs.main import (
    Main_Checkpoint_MiniMaxH3_Config,
    Main_Diffusers_MiniMaxH3_Config,
)
from invokeai.backend.model_manager.taxonomy import BaseModelType, MiniMaxH3VariantType, ModelType


def _identifier(key: str) -> ModelIdentifierField:
    return ModelIdentifierField(key=key, hash="h", name=key, base=BaseModelType.MiniMaxH3, type=ModelType.Main)


def _checkpoint(name: str, variant: MiniMaxH3VariantType, pruned: bool) -> Main_Checkpoint_MiniMaxH3_Config:
    return Main_Checkpoint_MiniMaxH3_Config.model_construct(name=name, variant=variant, pruned=pruned, path=name)


def _context(configs: dict[str, object]) -> MagicMock:
    context = MagicMock()
    context.models.exists.side_effect = lambda key: key in configs
    context.models.get_config.side_effect = lambda key: configs[key]
    return context


FL2VA_PRUNED = _checkpoint("fl2va pruned", MiniMaxH3VariantType.FL2VA, pruned=True)
REF2VA_PRUNED = _checkpoint("ref2va pruned", MiniMaxH3VariantType.REF2VA, pruned=True)
REF2VA_FULL = _checkpoint("ref2va full", MiniMaxH3VariantType.REF2VA, pruned=False)


def _node(transformer_key: str = "fl2va", overlay_key: str = "ref2va", **kwargs) -> MiniMaxH3HybridOverlayInvocation:
    return MiniMaxH3HybridOverlayInvocation(
        id="hybrid",
        transformer=MiniMaxH3TransformerField(
            transformer=_identifier(transformer_key),
            variant="fl2va",
            loras=[LoRAField(lora=_identifier("turbo"), weight=1.0)],
        ),
        overlay_model=_identifier(overlay_key),
        **kwargs,
    )


def test_records_the_overlay_and_stamps_the_overlay_task():
    output = _node(start_block=30, end_block=49, include_final_layer=True).invoke(
        _context({"fl2va": FL2VA_PRUNED, "ref2va": REF2VA_PRUNED})
    )
    field = output.transformer
    assert field.transformer.key == "fl2va"
    assert field.adaln_overlay is not None
    assert field.adaln_overlay.overlay.key == "ref2va"
    assert (field.adaln_overlay.start_block, field.adaln_overlay.end_block) == (30, 49)
    assert field.adaln_overlay.include_final_layer is True
    # The AdaLN projections decide the task: the denoise guard must now expect references.
    assert field.variant == "ref2va"
    # Earlier LoRAs ride along; the echo lets a graph record the overlay.
    assert [lora.lora.key for lora in field.loras] == ["turbo"]
    assert output.overlay_model.key == "ref2va"


def test_accepts_a_full_diffusers_base_with_a_full_overlay():
    diffusers = Main_Diffusers_MiniMaxH3_Config.model_construct(name="H3 folder", variant=MiniMaxH3VariantType.FL2VA)
    output = _node().invoke(_context({"fl2va": diffusers, "ref2va": REF2VA_FULL}))
    assert output.transformer.adaln_overlay is not None


def test_rejects_pruned_overlay_on_full_base_and_vice_versa():
    diffusers = Main_Diffusers_MiniMaxH3_Config.model_construct(name="H3 folder", variant=MiniMaxH3VariantType.FL2VA)
    with pytest.raises(ValueError, match="AdaLN-pruned.*full"):
        _node().invoke(_context({"fl2va": diffusers, "ref2va": REF2VA_PRUNED}))
    with pytest.raises(ValueError, match="full.*AdaLN-pruned"):
        _node().invoke(_context({"fl2va": FL2VA_PRUNED, "ref2va": REF2VA_FULL}))


def test_rejects_an_overlay_that_is_not_an_h3_checkpoint():
    lora = LoRA_LyCORIS_MiniMaxH3_Config.model_construct(name="lora", type=ModelType.LoRA)
    with pytest.raises(ValueError, match="not a MiniMax H3 single-file transformer checkpoint"):
        _node().invoke(_context({"fl2va": FL2VA_PRUNED, "ref2va": lora}))
    with pytest.raises(ValueError, match="Unknown overlay model"):
        _node().invoke(_context({"fl2va": FL2VA_PRUNED}))


def test_rejects_an_empty_selection_and_the_base_as_its_own_overlay():
    with pytest.raises(ValueError, match="selects nothing"):
        _node(start_block=40, end_block=30).invoke(_context({"fl2va": FL2VA_PRUNED, "ref2va": REF2VA_PRUNED}))
    with pytest.raises(ValueError, match="overlay model is the transformer itself"):
        _node(overlay_key="fl2va").invoke(_context({"fl2va": FL2VA_PRUNED}))


def test_rejects_stacking_a_second_overlay():
    first = _node().invoke(_context({"fl2va": FL2VA_PRUNED, "ref2va": REF2VA_PRUNED}))
    second = MiniMaxH3HybridOverlayInvocation(
        id="again", transformer=first.transformer, overlay_model=_identifier("ref2va")
    )
    with pytest.raises(ValueError, match="already carries"):
        second.invoke(_context({"fl2va": FL2VA_PRUNED, "ref2va": REF2VA_PRUNED}))
