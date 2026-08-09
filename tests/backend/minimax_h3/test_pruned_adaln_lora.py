"""Run-time AdaLN LoRA re-injection on AdaLN-pruned MiniMax H3 transformers.

The delta is a pure function of the row timestep: ``delta(t) = up @ (down @ silu_temb(t))``.
These tests verify the grid interpolation semantics (identical to the pruned model's own
``adaln_t_table`` lerp), the hook plumbing (shared per-forward rows + per-module additive
delta, including multiple stacked LoRAs), full cleanup on exit, and the validation errors.
"""

import pytest
import torch

from invokeai.backend.minimax_h3.pruned_adaln_lora import (
    MINIMAX_H3_TIME_EMBED_DIM,
    MiniMaxH3SiluTembGrid,
    _interp_rows,
    apply_minimax_h3_pruned_adaln_lora_patches,
)
from invokeai.backend.patches.layers.lora_layer import LoRALayer

GRID_ROWS = 9
CURVE_DIM = 4
OUT_FEATURES = 10
RANK = 2


class _StubPrunedTransformer(torch.nn.Module):
    """Mimics the pruned transformer's shape: AdaLN linears fed a curve temb, with the
    forward receiving the distinct row timesteps as a `timestep` kwarg."""

    def __init__(self) -> None:
        super().__init__()
        block = torch.nn.Module()
        adaln = torch.nn.Module()
        adaln.linear = torch.nn.Linear(CURVE_DIM, OUT_FEATURES, bias=False)
        block.adaln_proj = adaln
        self.transformer_blocks = torch.nn.ModuleList([block])
        norm_out = torch.nn.Module()
        norm_out.linear = torch.nn.Linear(CURVE_DIM, OUT_FEATURES, bias=False)
        self.norm_out = norm_out

    def forward(self, curve_temb: torch.Tensor, timestep: torch.Tensor | None = None):
        return (
            self.transformer_blocks[0].adaln_proj.linear(curve_temb),
            self.norm_out.linear(curve_temb),
        )


def _make_grid() -> torch.Tensor:
    torch.manual_seed(3)
    return torch.randn(GRID_ROWS, MINIMAX_H3_TIME_EMBED_DIM)


def _make_lora(seed: int) -> LoRALayer:
    torch.manual_seed(seed)
    return LoRALayer(
        up=torch.randn(OUT_FEATURES, RANK),
        mid=None,
        down=torch.randn(RANK, MINIMAX_H3_TIME_EMBED_DIM),
        alpha=None,
        bias=None,
    )


def _expected_delta(grid: torch.Tensor, layer: LoRALayer, weight: float, t: torch.Tensor) -> torch.Tensor:
    silu_temb = _interp_rows(grid, t)
    return weight * (silu_temb @ layer.down.T) @ layer.up.T


def test_interp_rows_matches_pruned_curve_semantics():
    grid = _make_grid()
    # Exact grid points, including both endpoints (t=1.0 must read the last row, not overflow).
    t = torch.tensor([0.0, 1.0, 4 / (GRID_ROWS - 1)])
    rows = _interp_rows(grid, t)
    assert torch.allclose(rows[0], grid[0])
    assert torch.allclose(rows[1], grid[-1])
    assert torch.allclose(rows[2], grid[4])
    # Midpoint between rows 2 and 3.
    t_mid = torch.tensor([2.5 / (GRID_ROWS - 1)])
    assert torch.allclose(_interp_rows(grid, t_mid)[0], (grid[2] + grid[3]) / 2, atol=1e-6)
    # Out-of-range clamps (the keyframe timestep 0.999 stays in range by design).
    assert torch.allclose(_interp_rows(grid, torch.tensor([1.5]))[0], grid[-1])
    assert torch.allclose(_interp_rows(grid, torch.tensor([-0.5]))[0], grid[0])


def test_injection_adds_expected_delta_and_cleans_up():
    transformer = _StubPrunedTransformer()
    grid = _make_grid()
    block_layer = _make_lora(seed=10)
    out_layer = _make_lora(seed=11)

    curve_temb = torch.randn(2, CURVE_DIM)
    timestep = torch.tensor([0.25, 0.999])
    base_block, base_out = transformer(curve_temb, timestep=timestep)

    patches = [
        ("transformer_blocks.0.adaln_proj.linear", block_layer, 0.5),
        ("norm_out.linear", out_layer, 1.0),
    ]
    with apply_minimax_h3_pruned_adaln_lora_patches(transformer, patches, grid):
        got_block, got_out = transformer(curve_temb, timestep=timestep)

    assert torch.allclose(got_block, base_block + _expected_delta(grid, block_layer, 0.5, timestep), atol=1e-5)
    assert torch.allclose(got_out, base_out + _expected_delta(grid, out_layer, 1.0, timestep), atol=1e-5)

    # Hooks removed: the module behaves exactly as before.
    after_block, after_out = transformer(curve_temb, timestep=timestep)
    assert torch.equal(after_block, base_block)
    assert torch.equal(after_out, base_out)
    assert not transformer._forward_pre_hooks
    assert not transformer.norm_out.linear._forward_hooks


def test_multiple_loras_stack_additively():
    transformer = _StubPrunedTransformer()
    grid = _make_grid()
    layer_a = _make_lora(seed=20)
    layer_b = _make_lora(seed=21)

    curve_temb = torch.randn(1, CURVE_DIM)
    timestep = torch.tensor([0.7])
    base_block, _ = transformer(curve_temb, timestep=timestep)

    patches = [
        ("transformer_blocks.0.adaln_proj.linear", layer_a, 1.0),
        ("transformer_blocks.0.adaln_proj.linear", layer_b, 0.25),
    ]
    with apply_minimax_h3_pruned_adaln_lora_patches(transformer, patches, grid):
        got_block, _ = transformer(curve_temb, timestep=timestep)

    expected = (
        base_block + _expected_delta(grid, layer_a, 1.0, timestep) + _expected_delta(grid, layer_b, 0.25, timestep)
    )
    assert torch.allclose(got_block, expected, atol=1e-5)


def test_missing_timestep_fails_loudly():
    transformer = _StubPrunedTransformer()
    patches = [("norm_out.linear", _make_lora(seed=30), 1.0)]
    with apply_minimax_h3_pruned_adaln_lora_patches(transformer, patches, _make_grid()):
        with pytest.raises(RuntimeError, match="did not receive a `timestep`"):
            transformer(torch.randn(1, CURVE_DIM))


def test_rejects_wrong_input_dim():
    transformer = _StubPrunedTransformer()
    bad_layer = LoRALayer(
        up=torch.randn(OUT_FEATURES, RANK), mid=None, down=torch.randn(RANK, 64), alpha=None, bias=None
    )
    with pytest.raises(ValueError, match="does not target MiniMax H3"):
        with apply_minimax_h3_pruned_adaln_lora_patches(
            transformer, [("norm_out.linear", bad_layer, 1.0)], _make_grid()
        ):
            pass


def test_rejects_output_dim_mismatch():
    transformer = _StubPrunedTransformer()
    bad_layer = LoRALayer(
        up=torch.randn(OUT_FEATURES + 1, RANK),
        mid=None,
        down=torch.randn(RANK, MINIMAX_H3_TIME_EMBED_DIM),
        alpha=None,
        bias=None,
    )
    with pytest.raises(ValueError, match="produces .* outputs"):
        with apply_minimax_h3_pruned_adaln_lora_patches(
            transformer, [("norm_out.linear", bad_layer, 1.0)], _make_grid()
        ):
            pass


def test_grid_loader_validates(tmp_path):
    from safetensors.torch import save_file

    good = tmp_path / "grid.safetensors"
    save_file({"silu_t_emb_grid": torch.randn(GRID_ROWS, MINIMAX_H3_TIME_EMBED_DIM, dtype=torch.float32)}, str(good))
    grid_module = MiniMaxH3SiluTembGrid.load_model(good)
    assert grid_module.grid.shape == (GRID_ROWS, MINIMAX_H3_TIME_EMBED_DIM)
    assert grid_module.grid.dtype == torch.float32

    missing = tmp_path / "missing.safetensors"
    save_file({"something_else": torch.zeros(2, 2)}, str(missing))
    with pytest.raises(ValueError, match="missing"):
        MiniMaxH3SiluTembGrid.load_model(missing)

    wrong_shape = tmp_path / "wrong.safetensors"
    save_file({"silu_t_emb_grid": torch.zeros(GRID_ROWS, 64)}, str(wrong_shape))
    with pytest.raises(ValueError, match="expected"):
        MiniMaxH3SiluTembGrid.load_model(wrong_shape)
