"""Hybrid AdaLN overlay for MiniMax H3: Ref2VA time-conditioning projections on an FL2VA base.

MiniMax shipped two H3 task transformers with an identical weight layout. FL2VA (first/last
keyframe conditioning) produces noticeably better output; Ref2VA additionally routes image,
video and audio references through the model, but carries a training defect that degrades its
raw output. A tensor-by-tensor comparison of the two (scottmudge/ComfyUI_MinimaxH3HybridLoader,
``minimax_h3_analysis.md``) shows that >97% of the weights - attention, MLPs, norms, patch
projections, token refiner - are near-identical (cosine >= 0.9997), and that the checkpoints
only diverge meaningfully in the per-block AdaLN modulation projections
(``blocks.N.adaln_proj.linear``, cosine ~ -0.8: rewritten wholesale to teach the
reference-modality mixture), the final-layer AdaLN projection (the single most discordant
tensor) and, mildly, the output heads.

The hybrid therefore runs the FL2VA transformer with the Ref2VA AdaLN projections swapped in
for a range of blocks - the reference recommends the upper half, 25..49, as the balance between
FL2VA quality and reference adherence - keeping FL2VA's attention, MLPs and output heads. The
swap happens at denoise time on the cached FL2VA transformer, the same shape as LoRA patching:
no merged checkpoint is written, and the base model stays shared in the model cache.

Only the selected AdaLN tensors are read from the overlay checkpoint (mmap): ~1.7 MB per block
on the AdaLN-pruned repacks, ~520 MB per block on a full bf16 checkpoint. The model cache keys
local models by path alone, so the selection rides on the path as a fragment and each distinct
selection is its own cache record.

On the AdaLN-pruned repacks the projections are coefficients over each file's OWN rank-8
time-embedding curve (``adaln_t_table``), and the two files' tables differ (several columns are
sign-flipped). A raw tensor swap - what the reference loader does - therefore reproduces only
part of Ref2VA's modulation. The tables span the same space, so the overlay re-expresses the
coefficients in the base's basis (an 8x8 least-squares change of basis) and reproduces Ref2VA's
modulation on the base's curve to float precision.
"""

import re
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch
from safetensors import safe_open

from invokeai.backend.util import InvokeAILogger
from invokeai.backend.util.original_weights_storage import OriginalWeightsStorage

# Checkpoint (remote-code layout) keys of the AdaLN projections, and the vendored diffusers-layout
# parameter paths they map onto (the same renames ``minimax_h3_state_dict_utils`` applies).
_BLOCK_ADALN_KEY = "blocks.{block}.adaln_proj.linear."
_FINAL_ADALN_KEY = "final_layer.adaln_proj.linear."
_CURVE_TABLE_KEY = "adaln_t_table"
_BLOCK_ADALN_PARAM = "transformer_blocks.{block}.adaln_proj.linear."
_FINAL_ADALN_PARAM = "norm_out.linear."
_PARAM_NAMES = ("weight", "bias")

_SELECTION_FRAGMENT = "#adaln-"
_SELECTION_RE = re.compile(r"^(\d+)-(\d+)(-final)?$")
_BLOCK_KEY_RE = re.compile(r"^blocks\.(\d+)\.adaln_proj\.linear\.weight$")

# Relative least-squares residual above which the overlay's curve is not a rotation of the base's
# and the re-expressed coefficients only approximate the overlay's modulation.
_CURVE_BASIS_RESIDUAL_WARN = 1e-3


class MiniMaxH3AdaLNOverlay(torch.nn.Module):
    """One selection of AdaLN projection tensors from an H3 single-file transformer, held as
    buffers so the model cache can size and pin it like any other model.

    Tensors keep the dtype the file stores (f16 on the AdaLN-pruned repacks, bf16 on full
    checkpoints); they are cast to the target parameter's dtype when applied.
    """

    def __init__(
        self,
        blocks: dict[int, dict[str, torch.Tensor]],
        final_layer: dict[str, torch.Tensor] | None,
        curve_table: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.blocks = tuple(sorted(blocks))
        self.has_final_layer = final_layer is not None
        for block, tensors in blocks.items():
            for name in _PARAM_NAMES:
                self.register_buffer(f"block_{block}_{name}", tensors[name])
        if final_layer is not None:
            for name in _PARAM_NAMES:
                self.register_buffer(f"final_{name}", final_layer[name])
        self.curve_table: torch.Tensor | None
        self.register_buffer("curve_table", curve_table)

    @staticmethod
    def selection_path(path: Path, start_block: int, end_block: int, include_final_layer: bool) -> Path:
        """The cache path of one selection of ``path``: the file path plus a selection fragment."""
        fragment = f"{_SELECTION_FRAGMENT}{start_block}-{end_block}{'-final' if include_final_layer else ''}"
        return path.with_name(path.name + fragment)

    @staticmethod
    def _parse_selection_path(path: Path) -> tuple[Path, int, int, bool]:
        name, sep, fragment = path.name.rpartition(_SELECTION_FRAGMENT)
        match = _SELECTION_RE.match(fragment) if sep else None
        if match is None:
            raise ValueError(f"Not an AdaLN overlay selection path (expected '<file>{_SELECTION_FRAGMENT}...'): {path}")
        return path.with_name(name), int(match.group(1)), int(match.group(2)), match.group(3) is not None

    @classmethod
    def load_model(cls, path: Path) -> "MiniMaxH3AdaLNOverlay":
        """Loader for ``context.models.load_local_model``; ``path`` comes from ``selection_path``.

        Reads only the selected AdaLN projections (blocks clamped to the ones the file has) plus
        the file's curve table, if it is an AdaLN-pruned checkpoint.
        """
        file_path, start_block, end_block, include_final_layer = cls._parse_selection_path(path)
        with safe_open(file_path, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            file_blocks = [int(m.group(1)) for key in keys if (m := _BLOCK_KEY_RE.match(key))]
            if not file_blocks or _FINAL_ADALN_KEY + "weight" not in keys:
                raise ValueError(
                    f"{file_path.name} has no MiniMax H3 AdaLN projections (blocks.N.adaln_proj.linear / "
                    "final_layer.adaln_proj.linear); it is not an H3 transformer checkpoint."
                )
            quantized = sorted(key for key in keys if key.endswith(".adaln_proj.linear.comfy_quant"))
            if quantized:
                raise ValueError(
                    f"{file_path.name} quantizes its AdaLN projections ({quantized[0]}, ...), which the hybrid "
                    "AdaLN overlay cannot apply. Every released H3 repack keeps them unquantized."
                )
            num_blocks = max(file_blocks) + 1
            selected_blocks = range(max(start_block, 0), min(end_block, num_blocks - 1) + 1)
            if not selected_blocks and not include_final_layer:
                raise ValueError(
                    f"The hybrid AdaLN overlay selects nothing: blocks {start_block}..{end_block} of the "
                    f"{num_blocks}-block {file_path.name}, final layer excluded."
                )

            def read(prefix: str) -> dict[str, torch.Tensor]:
                missing = [name for name in _PARAM_NAMES if prefix + name not in keys]
                if missing:
                    raise ValueError(f"{file_path.name} is missing {prefix}{missing[0]}; the checkpoint is incomplete.")
                return {name: f.get_tensor(prefix + name) for name in _PARAM_NAMES}

            blocks = {block: read(_BLOCK_ADALN_KEY.format(block=block)) for block in selected_blocks}
            final_layer = read(_FINAL_ADALN_KEY) if include_final_layer else None
            curve_table = f.get_tensor(_CURVE_TABLE_KEY) if _CURVE_TABLE_KEY in keys else None
        return cls(blocks, final_layer, curve_table)

    def tensors_for(self, base_curve_table: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """The overlay tensors keyed by the vendored transformer's parameter paths.

        ``base_curve_table`` is the target transformer's ``adaln_t_table`` (``None`` for a full
        model). When both sides are AdaLN-pruned and their tables differ, the projection weights
        are re-expressed in the base's curve basis so the base's curve drives them to the same
        modulation the overlay's own curve would (see the module docstring).
        """
        weights: dict[str, torch.Tensor] = {}
        biases: dict[str, torch.Tensor] = {}
        for block in self.blocks:
            prefix = _BLOCK_ADALN_PARAM.format(block=block)
            weights[prefix + "weight"] = getattr(self, f"block_{block}_weight")
            biases[prefix + "bias"] = getattr(self, f"block_{block}_bias")
        if self.has_final_layer:
            weights[_FINAL_ADALN_PARAM + "weight"] = self.final_weight
            biases[_FINAL_ADALN_PARAM + "bias"] = self.final_bias

        if base_curve_table is not None and self.curve_table is not None:
            base_table = base_curve_table.detach().to(device="cpu", dtype=torch.float32)
            if not torch.equal(base_table, self.curve_table.to(torch.float32)):
                change = _curve_basis_change(base_table, self.curve_table)
                weights = {key: weight.to(torch.float32) @ change.T for key, weight in weights.items()}
        return {**weights, **biases}


def _curve_basis_change(base_table: torch.Tensor, overlay_table: torch.Tensor) -> torch.Tensor:
    """``P`` with ``base_table @ P ~= overlay_table`` (least squares, float64).

    A pruned projection computes ``W @ c(t)`` on its own file's curve ``c``. With
    ``c_overlay(t) = c_base(t) @ P`` (interpolation is linear, so the row relation carries over),
    ``W_overlay @ c_overlay(t) = (W_overlay @ P^T) @ c_base(t)``: the re-expressed weight is
    ``W @ P^T``. Biases are basis-independent.
    """
    if base_table.shape != overlay_table.shape:
        raise ValueError(
            f"The overlay's AdaLN curve table has shape {list(overlay_table.shape)}, the base's "
            f"{list(base_table.shape)}; the two checkpoints are not the same kind."
        )
    base = base_table.to(torch.float64)
    overlay = overlay_table.to(device="cpu", dtype=torch.float64)
    change = torch.linalg.lstsq(base, overlay).solution
    residual = torch.linalg.norm(base @ change - overlay) / torch.linalg.norm(overlay)
    if residual > _CURVE_BASIS_RESIDUAL_WARN:
        InvokeAILogger.get_logger(__name__).warning(
            f"The hybrid AdaLN overlay's time-embedding curve is not a rotation of the base's (relative "
            f"residual {residual:.2e}); the overlaid modulation only approximates the overlay checkpoint's."
        )
    return change.to(torch.float32)


@contextmanager
def apply_minimax_h3_adaln_overlay(
    transformer: torch.nn.Module,
    overlay_params: dict[str, torch.Tensor],
    cached_weights: dict[str, torch.Tensor] | None = None,
) -> Iterator[None]:
    """Swap the named transformer parameters for the overlay tensors; restore them on exit.

    Args:
        transformer: the loaded (device-locked) H3 transformer.
        overlay_params: parameter path -> replacement tensor, from ``MiniMaxH3AdaLNOverlay.tensors_for``.
        cached_weights: the model cache's CPU copy of the weights, when it keeps one - restoring
            from it avoids holding a second copy of each original (the LayerPatcher convention).

    Parameters are reassigned out-of-place (``param.data = ...``), never written in place: the
    current tensor may be the cache's canonical CPU copy, which other devices' model instances
    read in multi-GPU mode and which the cache restores from after unlocking. A parameter the
    partial loader left on the CPU IS that canonical tensor: it is saved without copying and
    restored by re-pointing at it, so no second CPU copy is ever held. Every swapped parameter
    is restored on success, cancellation and error, including a failure part-way through the
    swap.
    """
    original_weights = OriginalWeightsStorage(cached_weights)
    try:
        for path, tensor in overlay_params.items():
            try:
                param = transformer.get_parameter(path)
            except AttributeError as e:
                raise ValueError(f"The loaded transformer has no parameter {path!r} to overlay.") from e
            if param.shape != tensor.shape:
                raise ValueError(
                    f"AdaLN overlay tensor {path!r} has shape {list(tensor.shape)}, but the loaded transformer's "
                    f"has shape {list(param.shape)}. The base transformer and the overlay checkpoint must both "
                    "be AdaLN-pruned or both be full checkpoints."
                )
            original_weights.save(path, param, copy=param.device.type != "cpu")
            param.data = tensor.to(device=param.device, dtype=param.dtype)
        yield
    finally:
        for path, weight in original_weights.get_changed_weights():
            param = transformer.get_parameter(path)
            param.data = weight.to(device=param.device, dtype=param.dtype)
