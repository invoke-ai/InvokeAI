"""FLUX.2 [dev] Fun-ControlNet-Union: the control branch alibaba-pai trained for the FLUX.2 [dev] transformer.

Reference: VideoX-Fun ``videox_fun/models/flux2_transformer2d_control.py``, ``pipeline_flux2_control.py`` and
``config/flux2/flux2_control.yaml``.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import accelerate
import torch
from diffusers.models.transformers.transformer_flux2 import Flux2TransformerBlock
from einops import rearrange
from torch import nn

from invokeai.backend.flux2.sampling_utils import pack_flux2
from invokeai.backend.fun_controlnet.branch import BlockRunner, FunControlBranch
from invokeai.backend.fun_controlnet.conditioning import Encoder, build_fun_control_context

# VideoX-Fun `config/flux2/flux2_control.yaml`: four control blocks, their hints added after these double-stream base
# blocks. The checkpoint carries no config, so the map is a constant of the family.
FLUX2_FUN_CONTROL_LAYERS: tuple[int, ...] = (0, 2, 4, 6)
# 65 latent channels -- control latents (32), inpaint keep mask (1), masked-image latents (32) -- packed 2x2.
FLUX2_FUN_CONTROL_IN_DIM = 260
# FLUX.2 attention heads are 128 wide in every variant. The checkpoint does not say, so the head count follows from
# the width.
FLUX2_ATTENTION_HEAD_DIM = 128

# The file names the blocks after VideoX-Fun's attribute; the shared branch calls them `control_blocks`.
_CHECKPOINT_BLOCK_PREFIX = "control_transformer_blocks."
_BRANCH_BLOCK_PREFIX = "control_blocks."

_IMAGE_STREAM = "hidden_states"
_TEXT_STREAM = "encoder_hidden_states"
_ATTENTION_KWARGS = "joint_attention_kwargs"
# What diffusers adds to the attention kwargs when the transformer runs with its reference KV cache. Invoke never sets
# `kv_cache_mode`; were it set, the control blocks would read or write the base blocks' cache layers.
_KV_CACHE_KWARGS = ("kv_cache", "kv_cache_mode", "num_ref_tokens")


class Flux2FunControlBlock(Flux2TransformerBlock):
    """A FLUX.2 double-stream block plus the two projections of the control recipe; ``forward`` is the base block's.

    The block has no modulation of its own: FLUX.2 computes it once per forward and hands it to every block, and the
    control blocks take the base's.
    """

    def __init__(
        self, *, dim: int, num_attention_heads: int, attention_head_dim: int, mlp_ratio: float, first: bool
    ) -> None:
        super().__init__(
            dim=dim, num_attention_heads=num_attention_heads, attention_head_dim=attention_head_dim, mlp_ratio=mlp_ratio
        )
        if first:
            self.before_proj = nn.Linear(dim, dim)
        self.after_proj = nn.Linear(dim, dim)


class Flux2FunControlBranch(FunControlBranch):
    """The branch for ``Flux2Transformer2DModel``: control blocks thread their own text stream beside the base's.

    The hints land on the double-stream blocks' image stream, all of it: with reference images that stream carries
    their tokens after the generated ones, and the denoise pads the control context with zeros for them, as VideoX-Fun
    does.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 3.0,
        control_in_dim: int = FLUX2_FUN_CONTROL_IN_DIM,
        control_layers: Sequence[int] = FLUX2_FUN_CONTROL_LAYERS,
    ) -> None:
        blocks = [
            Flux2FunControlBlock(
                dim=hidden_size,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                mlp_ratio=mlp_ratio,
                first=index == 0,
            )
            for index in range(len(control_layers))
        ]
        super().__init__(nn.Linear(control_in_dim, hidden_size), blocks, control_layers)

    @classmethod
    def from_state_dict(
        cls,
        sd: Mapping[str, torch.Tensor],
        *,
        dtype: torch.dtype,
        control_layers: Sequence[int] = FLUX2_FUN_CONTROL_LAYERS,
        attention_head_dim: int = FLUX2_ATTENTION_HEAD_DIM,
    ) -> "Flux2FunControlBranch":
        """Build the branch the checkpoint describes and load it, cast to ``dtype``.

        The geometry is read off the tensors: the width and input features from ``control_img_in.weight``, the MLP
        ratio from block 0's ``ff.linear_in`` (a fused SwiGLU, twice the hidden MLP width), the block count from the
        highest block index. Anything that is not the FLUX.2 recipe is refused by name rather than loaded into a branch
        that would run and produce noise.
        """
        sd = {
            _BRANCH_BLOCK_PREFIX + key.removeprefix(_CHECKPOINT_BLOCK_PREFIX)
            if key.startswith(_CHECKPOINT_BLOCK_PREFIX)
            else key: tensor
            for key, tensor in sd.items()
        }
        weight = sd.get("control_img_in.weight")
        if weight is None or weight.dim() != 2:
            raise ValueError("Not a FLUX.2 Fun-ControlNet checkpoint: `control_img_in.weight` is missing.")
        hidden_size, control_in_dim = weight.shape
        if control_in_dim != FLUX2_FUN_CONTROL_IN_DIM:
            raise ValueError(
                f"FLUX.2 Fun-ControlNet takes {FLUX2_FUN_CONTROL_IN_DIM} control features per token (65 latent channels "
                f"packed 2x2); this checkpoint's `control_img_in` takes {control_in_dim}."
            )
        if hidden_size % attention_head_dim:
            raise ValueError(
                f"A {hidden_size}-wide branch cannot be split into {attention_head_dim}-wide attention heads."
            )
        linear_in = sd.get(f"{_BRANCH_BLOCK_PREFIX}0.ff.linear_in.weight")
        if linear_in is None or linear_in.dim() != 2 or linear_in.shape[0] % 2 or linear_in.shape[1] != hidden_size:
            raise ValueError(
                "This checkpoint has no FLUX.2 feed-forward in its first control block: "
                f"`{_CHECKPOINT_BLOCK_PREFIX}0.ff.linear_in.weight` must be (2 x MLP width, {hidden_size})."
            )
        mlp_ratio = linear_in.shape[0] // 2 / hidden_size
        layers = tuple(control_layers)
        block_indices = {
            int(parts[1])
            for key in sd
            if (parts := key.split(".")) and parts[0] == _BRANCH_BLOCK_PREFIX[:-1] and parts[1].isdigit()
        }
        num_blocks = max(block_indices, default=-1) + 1
        if num_blocks != len(layers):
            raise ValueError(
                f"This checkpoint has {num_blocks} control blocks; the FLUX.2 recipe places {len(layers)} after base "
                f"layers {layers}."
            )

        with accelerate.init_empty_weights():
            branch = cls(
                hidden_size=hidden_size,
                num_attention_heads=hidden_size // attention_head_dim,
                attention_head_dim=attention_head_dim,
                mlp_ratio=mlp_ratio,
                control_in_dim=control_in_dim,
                control_layers=layers,
            )
        missing, unexpected = branch.load_state_dict(sd, strict=False, assign=True)
        if missing or unexpected:
            raise ValueError(
                "This checkpoint does not match the FLUX.2 control block layout: "
                f"missing {sorted(missing)[:5]}, unexpected {sorted(unexpected)[:5]}."
            )
        return branch.to(dtype=dtype).eval()

    def base_blocks(self, transformer: nn.Module) -> Sequence[nn.Module]:
        return transformer.transformer_blocks

    def base_hidden_size(self, transformer: nn.Module) -> int:
        return transformer.inner_dim

    def image_stream(self, block_inputs: Mapping[str, Any]) -> torch.Tensor:
        return block_inputs[_IMAGE_STREAM]

    def block_runner(self, block_inputs: Mapping[str, Any]) -> BlockRunner:
        attention_kwargs = block_inputs.get(_ATTENTION_KWARGS) or {}
        if cache_kwargs := sorted(name for name in _KV_CACHE_KWARGS if name in attention_kwargs):
            raise ValueError(
                f"The FLUX.2 ControlNet cannot run beside the transformer's reference KV cache ({cache_kwargs}): its "
                "blocks would share the base blocks' cache layers."
            )
        # Modulation, RoPE, attention kwargs: the control blocks take exactly what the base block took. The chain runs
        # in the branch's dtype end to end, so the floating inputs (the two modulations) are cast on the way in, like
        # `c` and the text stream, and the hint on the way out (`add_hint`). RoPE is a (cos, sin) pair the attention
        # applies in float32 whatever the activations are, and is handed on untouched like the attention kwargs.
        passthrough = {
            name: value.to(self.dtype) if isinstance(value, torch.Tensor) and value.is_floating_point() else value
            for name, value in block_inputs.items()
            if name not in (_IMAGE_STREAM, _TEXT_STREAM)
        }
        # The chain starts from the base's text stream as it enters block 0 and carries its own copy from then on.
        text_stream = block_inputs[_TEXT_STREAM].to(self.dtype)

        def run(block: nn.Module, c: torch.Tensor) -> torch.Tensor:
            nonlocal text_stream
            text_stream, c = block(hidden_states=c, encoder_hidden_states=text_stream, **passthrough)
            return c

        return run

    def add_hint(self, output: Any, hint: torch.Tensor, scale: float) -> Any:
        encoder_hidden_states, hidden_states = output
        return encoder_hidden_states, hidden_states + (hint * scale).to(hidden_states.dtype)


def build_flux2_control_context(
    *,
    encode: Encoder,
    bn_mean: torch.Tensor,
    bn_std: torch.Tensor,
    control_image: torch.Tensor | None = None,
    inpaint_image: torch.Tensor | None = None,
    inpaint_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """The packed control context ``(1, h/2 * w/2, 260)``: ``[control (128) | keep mask (4) | masked image (128)]``.

    ``encode`` is the FLUX.2 VAE's raw encode, ``(1, 3, H, W)`` in [-1, 1] to the ``mode()`` latents
    ``(1, 32, H/8, W/8)``. VideoX-Fun normalizes an encoded image the way the transformer's latents are: patchified
    2x2 to 128 channels and batch-normalized with the VAE's running statistics (``bn_mean``, and ``bn_std`` =
    ``sqrt(running_var + eps)``, both ``(128,)``). Here the normalized latents are put back to the 32-channel layout,
    a permutation that changes no value, so the shared builder keeps the keep mask at latent resolution; packing then
    patchifies all 65 channels in one go, in the order VideoX-Fun concatenates its three patchified parts.
    """

    def encode_normalized(pixels: torch.Tensor) -> torch.Tensor:
        latents = encode(pixels)
        patches = rearrange(latents, "b c (h ph) (w pw) -> b (c ph pw) h w", ph=2, pw=2)
        mean = bn_mean.to(device=patches.device, dtype=patches.dtype).view(1, -1, 1, 1)
        std = bn_std.to(device=patches.device, dtype=patches.dtype).view(1, -1, 1, 1)
        return rearrange((patches - mean) / std, "b (c ph pw) h w -> b c (h ph) (w pw)", ph=2, pw=2)

    context = build_fun_control_context(
        encode=encode_normalized, control_image=control_image, inpaint_image=inpaint_image, inpaint_mask=inpaint_mask
    )
    return pack_flux2(context)
