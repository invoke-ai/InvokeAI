"""Per-call-site guards for the scale axis used when folding a scaled-fp8 weight.

A per-output-channel `weight_scale` is 1-D of length `out_features`, and `(out, in) * (out,)`
broadcasts on the *last* axis — so a bare multiply scales input channels instead of output
channels. That is a shape error on a non-square weight (loud) and a silently wrong weight on a
square one (not loud at all). Both loaders below used to do the bare multiply; they now go through
`expand_weight_scale`.

`test_fp8_scaled.py` covers the helper itself. These pin the two call sites, so reverting either
one back to its local loop fails a test rather than passing CI.
"""

from unittest.mock import MagicMock

import pytest
import torch

from invokeai.backend.model_manager.load.model_loaders.mistral_encoder import _drop_quantization_metadata
from invokeai.backend.model_manager.load.model_loaders.z_image import _fold_comfy_scaled_weights


def _per_channel_case(out_features: int = 4, in_features: int = 2):
    """A non-square weight with a per-output-channel scale: rows 0..n-1 scaled by 1..n."""
    weight = torch.ones(out_features, in_features).to(torch.float8_e4m3fn)
    scale = torch.arange(1, out_features + 1, dtype=torch.float32)
    expected = torch.arange(1, out_features + 1, dtype=torch.bfloat16).reshape(-1, 1).expand(-1, in_features)
    return weight, scale, expected


class TestMistralEncoderFold:
    def test_per_channel_scale_multiplies_rows(self) -> None:
        weight, scale, expected = _per_channel_case()
        sd = {"layer.weight": weight, "layer.weight_scale": scale}

        _drop_quantization_metadata(sd, MagicMock(), target_dtype=torch.bfloat16)

        assert torch.equal(sd["layer.weight"], expected)
        assert "layer.weight_scale" not in sd

    def test_square_weight_is_not_silently_transposed(self) -> None:
        """The dangerous case: a square weight broadcasts happily on the wrong axis."""
        sd = {
            "layer.weight": torch.ones(3, 3).to(torch.float8_e4m3fn),
            "layer.weight_scale": torch.tensor([1.0, 2.0, 3.0]),
        }

        _drop_quantization_metadata(sd, MagicMock(), target_dtype=torch.bfloat16)

        rows = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], dtype=torch.bfloat16)
        assert torch.equal(sd["layer.weight"], rows), "scale must vary down the rows, not across them"


class TestZImageQwen3EncoderFold:
    """The Z-Image Qwen3 encoder's single-file loader folds the same way."""

    def _fold(self, sd: dict, dtype: torch.dtype = torch.bfloat16) -> dict:
        # The real call site. It was a loop inside `_load_from_singlefile` -- unreachable without a
        # checkpoint and a transformers model -- so it was extracted; re-inlining it as a local
        # multiply fails these tests instead of passing CI.
        _fold_comfy_scaled_weights(sd, dtype)
        return sd

    def test_per_channel_scale_multiplies_rows(self) -> None:
        weight, scale, expected = _per_channel_case()
        sd = self._fold({"layer.weight": weight, "layer.weight_scale": scale})
        assert torch.equal(sd["layer.weight"], expected)

    def test_non_square_weight_does_not_raise(self) -> None:
        weight, scale, _ = _per_channel_case(out_features=6, in_features=2)
        sd = self._fold({"layer.weight": weight, "layer.weight_scale": scale})
        assert sd["layer.weight"].shape == (6, 2)

    def test_block_wise_scale_is_expanded_rather_than_rejected(self) -> None:
        sd = self._fold(
            {
                "layer.weight": torch.ones(4, 2).to(torch.float8_e4m3fn),
                "layer.weight_scale": torch.tensor([[1.0], [2.0]]),  # one entry per 2-row block
            }
        )
        assert torch.equal(
            sd["layer.weight"],
            torch.tensor([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.0, 2.0]], dtype=torch.bfloat16),
        )

    def test_a_scale_matching_neither_layout_is_reported(self) -> None:
        with pytest.raises(ValueError, match="neither per-tensor nor per-output-channel"):
            self._fold(
                {
                    "layer.weight": torch.ones(4, 2).to(torch.float8_e4m3fn),
                    "layer.weight_scale": torch.full((3,), 2.0),
                }
            )
