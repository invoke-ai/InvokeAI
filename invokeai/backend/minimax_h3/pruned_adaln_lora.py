"""Run-time re-injection of AdaLN LoRA deltas on AdaLN-pruned MiniMax H3 transformers.

H3 LoRAs (e.g. the Turbo step-distillation LoRA) carry low-rank updates for the
per-block AdaLN time-conditioning projections, whose input space is the full model's
2688-dim ``silu(t_emb)``. The AdaLN-pruned checkpoints collapsed exactly that space
into a precomputed rank-8 curve (``adaln_t_table`` + 8-input AdaLN linears), so those
LoRA layers can be neither directly merged nor applied as sidecar patches — the
2688-dim input no longer exists anywhere in the pruned forward.

They can still be applied exactly: the AdaLN delta is a pure function of the row
timestep, ``delta(t) = lora_up @ (lora_down @ silu(t_emb(t)))``, and ``silu(t_emb(t))``
is available as a small precomputed grid over ``t in [0, 1]`` (published alongside the
Turbo LoRA's ComfyUI node, Apache 2.0; interpolation semantics identical to the pruned
model's own ``adaln_t_table`` lerp). This module re-injects the delta at run time:

- a forward pre-hook on the transformer interpolates ``silu(t_emb)`` rows for the
  step's distinct row timesteps (video, audio and — when keyframes are present — the
  fixed 0.999 conditioning timestep all appear in ``timestep``, so keyframe rows are
  covered automatically);
- a forward hook on each targeted AdaLN ``linear`` adds its delta to the projection
  output, which the curve AdaLN modules then chunk into modulation parameters as usual.

Hooks leave the module tree and its weights untouched, so the patched transformer can
live in the shared model cache; the context manager removes every hook on exit.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch

from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.util import InvokeAILogger

# The silu(t_emb) grid published with the Turbo LoRA's ComfyUI node (Apache 2.0),
# pinned to a commit for reproducibility. ~5.5 MB; fetched once via
# ``context.models.load_remote_model`` and cached like any other model file.
MINIMAX_H3_SILU_TEMB_GRID_URL = (
    "https://raw.githubusercontent.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/"
    "e7ad532857f2327feb56cf7729a9a76857a6799f/h3_silu_temb_grid.safetensors"
)

# time_embed_dim of the full (non-pruned) H3 transformer — the LoRA's input space.
MINIMAX_H3_TIME_EMBED_DIM = 2688

_GRID_TENSOR_NAME = "silu_t_emb_grid"


class MiniMaxH3SiluTembGrid(torch.nn.Module):
    """The full H3 transformer's ``silu(t_emb)`` curve sampled on a uniform t-grid.

    A (buffer-only) ``nn.Module`` so ``load_remote_model``'s cache can size and move it
    like any other model.
    """

    def __init__(self, grid: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("grid", grid)

    @classmethod
    def load_model(cls, path: Path) -> "MiniMaxH3SiluTembGrid":
        """Loader for ``context.models.load_remote_model``."""
        from safetensors.torch import load_file

        state_dict = load_file(path)
        if _GRID_TENSOR_NAME not in state_dict:
            raise ValueError(f"silu(t_emb) grid file is missing the {_GRID_TENSOR_NAME!r} tensor: {path}")
        grid = state_dict[_GRID_TENSOR_NAME].to(torch.float32)
        if grid.ndim != 2 or grid.shape[0] < 2 or grid.shape[1] != MINIMAX_H3_TIME_EMBED_DIM:
            raise ValueError(
                f"silu(t_emb) grid has shape {list(grid.shape)}; expected [>=2, {MINIMAX_H3_TIME_EMBED_DIM}]."
            )
        return cls(grid)


def _interp_rows(table: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Linearly interpolate table rows at ``t in [0, 1]``.

    Same clamp/lerp semantics as ``MiniMaxH3PrunedTransformer3DModel._curve_temb``, so the
    delta rides the exact grid convention the pruned model itself uses.
    """
    pos = t.to(device=table.device, dtype=torch.float32).clamp(0.0, 1.0) * (table.shape[0] - 1)
    i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
    return torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))


class _AdalnLoRADelta:
    """One AdaLN LoRA layer's delta factors, moved to the compute device on first use."""

    def __init__(self, layer: LoRALayer, patch_weight: float) -> None:
        self.scale = patch_weight * layer.scale()
        self._down = layer.down
        self._up = layer.up
        self._cached: tuple[torch.device, torch.Tensor, torch.Tensor] | None = None

    def tensors_for(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cached is None or self._cached[0] != device:
            self._cached = (
                device,
                self._down.to(device=device, dtype=torch.float32),
                self._up.to(device=device, dtype=torch.float32),
            )
        return self._cached[1], self._cached[2]


@contextmanager
def apply_minimax_h3_pruned_adaln_lora_patches(
    transformer: torch.nn.Module,
    adaln_patches: list[tuple[str, LoRALayer, float]],
    silu_temb_grid: torch.Tensor,
) -> Iterator[None]:
    """Re-inject AdaLN LoRA deltas on an AdaLN-pruned H3 transformer via forward hooks.

    Args:
        transformer: a ``MiniMaxH3PrunedTransformer3DModel`` (already on its compute device
            or cache-managed).
        adaln_patches: ``(layer_path, lora_layer, patch_weight)`` triples whose paths point
            at the pruned model's AdaLN linears (``transformer_blocks.N.adaln_proj.linear``
            or ``norm_out.linear``).
        silu_temb_grid: ``[grid, 2688]`` float tensor — see ``MiniMaxH3SiluTembGrid``.
    """
    if silu_temb_grid.ndim != 2 or silu_temb_grid.shape[1] != MINIMAX_H3_TIME_EMBED_DIM:
        raise ValueError(
            f"silu(t_emb) grid has shape {list(silu_temb_grid.shape)}; expected [>=2, {MINIMAX_H3_TIME_EMBED_DIM}]."
        )

    # One shared set of interpolated silu(t_emb) rows per transformer forward.
    shared: dict[str, torch.Tensor | None] = {"silu_temb": None}
    grid_cache: dict[torch.device, torch.Tensor] = {}

    def transformer_pre_hook(module: torch.nn.Module, args: tuple, kwargs: dict) -> None:
        timestep = kwargs.get("timestep")
        if timestep is None and len(args) > 3:
            timestep = args[3]
        if timestep is None:
            # Leave a poison marker so the linear hooks fail loudly instead of silently
            # applying a stale delta from the previous step.
            shared["silu_temb"] = None
            return
        grid = grid_cache.get(timestep.device)
        if grid is None:
            grid = silu_temb_grid.to(device=timestep.device, dtype=torch.float32)
            grid_cache[timestep.device] = grid
        shared["silu_temb"] = _interp_rows(grid, timestep)

    def make_linear_hook(layer_path: str, delta: _AdalnLoRADelta):
        def linear_hook(module: torch.nn.Module, args: tuple, output: torch.Tensor) -> torch.Tensor:
            silu_temb = shared["silu_temb"]
            if silu_temb is None:
                raise RuntimeError(
                    f"AdaLN LoRA injection for {layer_path!r} ran without silu(t_emb) rows — the "
                    "transformer forward did not receive a `timestep` argument."
                )
            down, up = delta.tensors_for(output.device)
            d = (silu_temb.to(output.device) @ down.T) @ up.T
            return output + (d * delta.scale).to(output.dtype)

        return linear_hook

    logger = InvokeAILogger.get_logger(__name__)
    handles: list[torch.utils.hooks.RemovableHandle] = []
    try:
        for layer_path, layer, patch_weight in adaln_patches:
            if layer.mid is not None or layer.bias is not None:
                raise ValueError(f"AdaLN LoRA layer {layer_path!r} has unsupported mid/bias tensors.")
            # Architecture mismatches (unresolvable path, wrong dims) warn and skip the layer —
            # the same policy the LayerPatcher applies to backbone layers, so a partially
            # incompatible LoRA degrades identically on both routes instead of hard-failing
            # only when the AdaLN half is the incompatible one.
            if layer.down.shape[1] != MINIMAX_H3_TIME_EMBED_DIM:
                logger.warning(
                    f"Skipping AdaLN LoRA layer '{layer_path}': it expects a {layer.down.shape[1]}-dim "
                    f"input, but the H3 silu(t_emb) space is {MINIMAX_H3_TIME_EMBED_DIM}-dim. This LoRA "
                    "may be incompatible with this model architecture."
                )
                continue
            try:
                linear = transformer.get_submodule(layer_path)
            except AttributeError:
                logger.warning(f"Failed to find module for AdaLN LoRA layer: {layer_path}")
                continue
            out_features = getattr(linear, "out_features", None)
            if out_features is not None and layer.up.shape[0] != out_features:
                logger.warning(
                    f"Skipping AdaLN LoRA layer '{layer_path}' due to shape mismatch: the target "
                    f"projection has {out_features} outputs, the LoRA produces {layer.up.shape[0]}. "
                    "This LoRA may be incompatible with this model architecture."
                )
                continue
            handles.append(
                linear.register_forward_hook(make_linear_hook(layer_path, _AdalnLoRADelta(layer, patch_weight)))
            )

        if handles:
            handles.append(transformer.register_forward_pre_hook(transformer_pre_hook, with_kwargs=True))

        yield
    finally:
        for handle in handles:
            handle.remove()
        shared["silu_temb"] = None
