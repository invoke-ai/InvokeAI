"""The denoise node's side of the hybrid AdaLN overlay: resolving the recorded overlay to the
tensors it swaps in, through the model cache's local-file loader."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import save_file

from invokeai.app.invocations.minimax_h3.minimax_h3_denoise import MiniMaxH3DenoiseInvocation
from invokeai.app.invocations.model import MiniMaxH3AdaLNOverlayField, MiniMaxH3TransformerField, ModelIdentifierField
from invokeai.backend.minimax_h3.adaln_overlay import MiniMaxH3AdaLNOverlay
from invokeai.backend.model_manager.configs.lora import LoRA_LyCORIS_MiniMaxH3_Config
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_MiniMaxH3_Config
from invokeai.backend.model_manager.taxonomy import BaseModelType, MiniMaxH3VariantType, ModelType


def _identifier(key: str) -> ModelIdentifierField:
    return ModelIdentifierField(key=key, hash="h", name=key, base=BaseModelType.MiniMaxH3, type=ModelType.Main)


def _field(start_block: int, end_block: int, include_final_layer: bool = False) -> MiniMaxH3TransformerField:
    return MiniMaxH3TransformerField(
        transformer=_identifier("fl2va"),
        variant="ref2va",
        adaln_overlay=MiniMaxH3AdaLNOverlayField(
            overlay=_identifier("ref2va"),
            start_block=start_block,
            end_block=end_block,
            include_final_layer=include_final_layer,
        ),
    )


@pytest.fixture
def overlay_file(tmp_path: Path) -> Path:
    # Three blocks' AdaLN projections plus the final layer, nothing else needed.
    sd: dict[str, torch.Tensor] = {}
    for i in range(3):
        sd[f"blocks.{i}.adaln_proj.linear.weight"] = torch.tensor([[float(i), float(i) + 1.0]] * 4)
        sd[f"blocks.{i}.adaln_proj.linear.bias"] = torch.full((4,), float(i))
    sd["final_layer.adaln_proj.linear.weight"] = torch.ones(2, 2)
    sd["final_layer.adaln_proj.linear.bias"] = torch.ones(2)
    sd["adaln_t_table"] = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    path = tmp_path / "ref2va.safetensors"
    save_file(sd, str(path))
    return path


def _context(overlay_file: Path, config: object) -> MagicMock:
    context = MagicMock()
    context.models.exists.return_value = True
    context.models.get_config.return_value = config
    context.models.get_absolute_path.return_value = overlay_file
    # The real loader runs against the real file; only the cache is stood in for.
    context.models.load_local_model.side_effect = lambda path, loader: SimpleNamespace(model=loader(path))
    return context


REF2VA = Main_Checkpoint_MiniMaxH3_Config.model_construct(
    name="ref2va", variant=MiniMaxH3VariantType.REF2VA, pruned=True, path="ref2va.safetensors"
)


def _materialize(field: MiniMaxH3TransformerField, context: MagicMock, transformer: object = SimpleNamespace()):
    return MiniMaxH3DenoiseInvocation._materialize_adaln_overlay(
        SimpleNamespace(transformer=field), context, transformer
    )


def test_no_overlay_is_a_no_op():
    context = MagicMock()
    field = MiniMaxH3TransformerField(transformer=_identifier("fl2va"))
    assert _materialize(field, context) is None
    context.models.load_local_model.assert_not_called()


def test_resolves_the_recorded_range_through_the_local_model_cache(overlay_file: Path):
    context = _context(overlay_file, REF2VA)
    info, params = _materialize(_field(1, 49), context)

    assert isinstance(info.model, MiniMaxH3AdaLNOverlay)
    # One cache record per selection: the range rides on the path the cache keys by.
    context.models.load_local_model.assert_called_once_with(
        MiniMaxH3AdaLNOverlay.selection_path(overlay_file, 1, 49, False), MiniMaxH3AdaLNOverlay.load_model
    )
    assert set(params) == {f"transformer_blocks.{i}.adaln_proj.linear.{p}" for i in (1, 2) for p in ("weight", "bias")}
    assert torch.equal(params["transformer_blocks.2.adaln_proj.linear.bias"], torch.full((4,), 2.0))


def test_refuses_an_overlay_model_that_is_not_an_h3_checkpoint(overlay_file: Path):
    lora = LoRA_LyCORIS_MiniMaxH3_Config.model_construct(name="lora", type=ModelType.LoRA)
    with pytest.raises(ValueError, match="not a MiniMax H3 single-file transformer checkpoint"):
        _materialize(_field(0, 49), _context(overlay_file, lora))


def test_refuses_a_range_beyond_the_files_blocks(overlay_file: Path):
    # The node validates against the released 50-block layout; the file is the authority.
    with pytest.raises(ValueError, match="selects nothing: blocks 10..49 of the 3-block"):
        _materialize(_field(10, 49), _context(overlay_file, REF2VA))
    _, params = _materialize(_field(10, 49, include_final_layer=True), _context(overlay_file, REF2VA))
    assert set(params) == {"norm_out.linear.weight", "norm_out.linear.bias"}


def test_re_expresses_the_weights_on_the_loaded_transformers_curve(overlay_file: Path):
    # The pruned transformer's own table is handed to the overlay: same table, same weights;
    # a rotated table, re-expressed weights (the re-expression itself is covered in the backend suite).
    same_table = SimpleNamespace(adaln_t_table=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    _, params = _materialize(_field(2, 2), _context(overlay_file, REF2VA), same_table)
    assert torch.equal(params["transformer_blocks.2.adaln_proj.linear.weight"], torch.tensor([[2.0, 3.0]] * 4))

    # The base's table has the file's two curve columns swapped, so the coefficients swap too.
    swapped_table = SimpleNamespace(adaln_t_table=torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]))
    _, params = _materialize(_field(2, 2), _context(overlay_file, REF2VA), swapped_table)
    assert torch.allclose(params["transformer_blocks.2.adaln_proj.linear.weight"], torch.tensor([[3.0, 2.0]] * 4))
    assert torch.equal(params["transformer_blocks.2.adaln_proj.linear.bias"], torch.full((4,), 2.0))
