"""Z-Image Fun-ControlNet: the control alibaba-pai trained for the Z-Image transformer, in the four layouts it published.

Reference: VideoX-Fun ``videox_fun/models/z_image_transformer2d_control.py``, ``pipeline_z_image_control.py`` and
``config/z_image/z_image_control{,_2.0,_2.1,_2.1_lite}.yaml``.

The chain is split across the base's own two stages, so it runs as two hooked stages:

- **Refiner stage**, entered at ``noise_refiner[0]`` on the image stream alone. The control context is embedded, padded
  the way the image stream is, and run through the refiner chain. From 2.0 on that chain joins the image stream like a
  VACE chain and adds its two hints after both base refiner blocks; 1.0 runs plain refiner blocks and adds nothing here.
- **Main stage**, entered at ``layers[0]`` on the unified ``[image | caption]`` stream. The refined control stream is
  joined with the caption part of that stream and runs through ``control_layers``, one hint per block after the listed
  base layers.

The checkpoints carry no config, and 2.0 and 2.1 have the same keys and shapes; the variant is read off the embedder
width, the block count and whether the refiner projections were trained (2.0's are all zero).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import accelerate
import torch
import torch.nn.functional as F
from diffusers.models.transformers.transformer_z_image import SEQ_MULTI_OF, ZImageTransformerBlock
from einops import rearrange
from torch import nn

from invokeai.backend.fun_controlnet.branch import BlockRunner, FunControl, FunControlStage, hint_blocks, run_vace_chain

# Z-Image attention heads are 128 wide. The checkpoint does not say, so the head count follows from the width.
Z_IMAGE_ATTENTION_HEAD_DIM = 128
# VideoX-Fun `control_refiner_layers_places`: the refiner chain's hints land after both base refiner blocks.
Z_IMAGE_FUN_CONTROL_REFINER_LAYERS: tuple[int, ...] = (0, 1)
_NUM_REFINER_BLOCKS = 2
# The transformer's only patch size, 2x2 spatial by 1 frame, names the embedder.
_EMBEDDER_KEY = "2-1"
_EMBEDDER_WEIGHT = f"control_all_x_embedder.{_EMBEDDER_KEY}.weight"
# Every Z-Image block takes its stream as `x`; the other arguments are handed to the control blocks as they are.
_STREAM = "x"
_REFINER_PROJECTIONS = tuple(
    f"control_noise_refiner.{block}.{projection}.{tensor}"
    for block, projection in ((0, "before_proj"), (0, "after_proj"), (1, "after_proj"))
    for tensor in ("weight", "bias")
)

RefinerChain = Literal["plain", "refiner", "main"]


@dataclass(frozen=True)
class ZImageFunControlVariant:
    """One published layout.

    ``refiner_chain`` is what runs in the refiner stage: ``plain`` refiner blocks with no projections, which neither
    join the image stream nor add hints (1.0); the ``refiner`` blocks with their projections (2.1, Lite); or the
    ``main`` ``control_layers`` (2.0, whose config ran the refiner chain over them; its own refiner blocks were never
    trained and are not built).
    """

    name: str
    control_in_channels: int
    main_layers: tuple[int, ...]
    refiner_chain: RefinerChain


Z_IMAGE_FUN_CONTROL_1_0 = ZImageFunControlVariant("1.0", 16, (0, 5, 10, 15, 20, 25), "plain")
Z_IMAGE_FUN_CONTROL_2_0 = ZImageFunControlVariant("2.0", 33, tuple(range(0, 30, 2)), "main")
Z_IMAGE_FUN_CONTROL_2_1 = ZImageFunControlVariant("2.1", 33, tuple(range(0, 30, 2)), "refiner")
Z_IMAGE_FUN_CONTROL_2_1_LITE = ZImageFunControlVariant("2.1 Lite", 33, (0, 10, 20), "refiner")

# The refiner projections each refiner chain leaves in the file: 1.0 has none; 2.0 carries them all zero, as initialised,
# because its config never ran them; 2.1 and Lite trained them.
_REFINER_PROJECTION_STATE: dict[RefinerChain, str] = {"plain": "absent", "main": "untrained", "refiner": "trained"}
# What tells the layouts apart: embedder input features (2x2 patches), control blocks, and the refiner projections.
_LAYOUTS: dict[tuple[int, int, str], ZImageFunControlVariant] = {
    (
        4 * variant.control_in_channels,
        len(variant.main_layers),
        _REFINER_PROJECTION_STATE[variant.refiner_chain],
    ): variant
    for variant in (
        Z_IMAGE_FUN_CONTROL_1_0,
        Z_IMAGE_FUN_CONTROL_2_0,
        Z_IMAGE_FUN_CONTROL_2_1,
        Z_IMAGE_FUN_CONTROL_2_1_LITE,
    )
}


class ZImageFunControlBlock(ZImageTransformerBlock):
    """A Z-Image block plus the projections of the control recipe it carries; ``forward`` is the base block's."""

    def __init__(
        self, *, layer_id: int, dim: int, num_attention_heads: int, before_proj: bool, after_proj: bool
    ) -> None:
        # Z-Image's own block settings: RMSNorm eps 1e-5, query/key norm, timestep modulation.
        super().__init__(
            layer_id,
            dim,
            n_heads=num_attention_heads,
            n_kv_heads=num_attention_heads,
            norm_eps=1e-5,
            qk_norm=True,
            modulation=True,
        )
        if before_proj:
            self.before_proj = nn.Linear(dim, dim)
        if after_proj:
            self.after_proj = nn.Linear(dim, dim)


class ZImageFunControl(FunControl):
    """The control for ``ZImageTransformer2DModel``, under the checkpoint's own names: ``control_all_x_embedder``,
    ``control_noise_refiner`` and ``control_layers``."""

    def __init__(self, variant: ZImageFunControlVariant, *, hidden_size: int, num_attention_heads: int) -> None:
        super().__init__()
        self.variant = variant
        self.control_all_x_embedder = nn.ModuleDict(
            {_EMBEDDER_KEY: nn.Linear(4 * variant.control_in_channels, hidden_size)}
        )
        self.control_layers = nn.ModuleList(
            ZImageFunControlBlock(
                layer_id=index,
                dim=hidden_size,
                num_attention_heads=num_attention_heads,
                before_proj=index == 0,
                after_proj=True,
            )
            for index in range(len(variant.main_layers))
        )
        joined = variant.refiner_chain == "refiner"
        self.control_noise_refiner = nn.ModuleList(
            ZImageFunControlBlock(
                layer_id=1000 + index,
                dim=hidden_size,
                num_attention_heads=num_attention_heads,
                before_proj=joined and index == 0,
                after_proj=joined,
            )
            for index in range(_NUM_REFINER_BLOCKS if variant.refiner_chain != "main" else 0)
        )

    @classmethod
    def from_state_dict(
        cls, sd: Mapping[str, torch.Tensor], *, dtype: torch.dtype, attention_head_dim: int = Z_IMAGE_ATTENTION_HEAD_DIM
    ) -> "ZImageFunControl":
        """Build the control the checkpoint describes and load it, cast to ``dtype``.

        The width comes off the embedder, the variant off the embedder's input features, the control block count and
        the refiner projections (2.0's are all zero). Anything that is not one of the published layouts is refused by
        name rather than loaded into a control that would run and produce noise.
        """
        weight = sd.get(_EMBEDDER_WEIGHT)
        if weight is None or weight.dim() != 2:
            raise ValueError(f"Not a Z-Image Fun-ControlNet checkpoint: `{_EMBEDDER_WEIGHT}` is missing.")
        hidden_size, in_features = weight.shape
        if hidden_size % attention_head_dim:
            raise ValueError(
                f"A {hidden_size}-wide control cannot be split into {attention_head_dim}-wide attention heads."
            )
        variant = _variant_of(sd, in_features)
        if variant.refiner_chain == "main":
            sd = {key: tensor for key, tensor in sd.items() if not key.startswith("control_noise_refiner.")}

        with accelerate.init_empty_weights():
            control = cls(variant, hidden_size=hidden_size, num_attention_heads=hidden_size // attention_head_dim)
        missing, unexpected = control.load_state_dict(dict(sd), strict=False, assign=True)
        if missing or unexpected:
            raise ValueError(
                f"This checkpoint does not match the Z-Image Fun-ControlNet {variant.name} layout: "
                f"missing {sorted(missing)[:5]}, unexpected {sorted(unexpected)[:5]}."
            )
        return control.to(dtype=dtype).eval()

    @property
    def hidden_size(self) -> int:
        return self._embedder.out_features

    @property
    def control_in_dim(self) -> int:
        return self._embedder.in_features

    @property
    def dtype(self) -> torch.dtype:
        return self._embedder.weight.dtype

    @property
    def _embedder(self) -> nn.Linear:
        return self.control_all_x_embedder[_EMBEDDER_KEY]

    def pack(self, context: torch.Tensor) -> torch.Tensor:
        """The control context ``(B, 33, h, w)`` as the tokens the embedder takes, ``(B, h/2 * w/2, 4 * C)``.

        Patched 2x2 in the order the transformer patches its latents (``_patchify_image``: row-major tokens, features
        in ``(pH, pW, C)`` order). The 1.0 layout takes the control latents alone, so only their 16 channels are kept.
        """
        return rearrange(
            context[:, : self.variant.control_in_channels], "b c (h ph) (w pw) -> b (h w) (ph pw c)", ph=2, pw=2
        )

    def base_hidden_size(self, transformer: nn.Module) -> int:
        return transformer.dim

    def stages(self, transformer: nn.Module, control_context: torch.Tensor) -> Sequence[FunControlStage]:
        refiner_targets = (
            hint_blocks(transformer.noise_refiner, Z_IMAGE_FUN_CONTROL_REFINER_LAYERS)
            if self.variant.refiner_chain != "plain"
            else ()
        )
        return (
            FunControlStage(
                entry=transformer.noise_refiner[0],
                hint_blocks=refiner_targets,
                run=partial(self._run_refiner_stage, control_context),
            ),
            FunControlStage(
                entry=transformer.layers[0],
                hint_blocks=hint_blocks(transformer.layers, self.variant.main_layers),
                run=self._run_main_stage,
            ),
        )

    def add_hint(self, output: torch.Tensor, hint: torch.Tensor, scale: float) -> torch.Tensor:
        return output + (hint * scale).to(output.dtype)

    def _run_refiner_stage(
        self, control_context: torch.Tensor, inputs: Mapping[str, Any], _carry: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        x = inputs[_STREAM]
        num_tokens, padded_tokens = control_context.shape[1], x.shape[1]
        if padded_tokens != num_tokens + (-num_tokens) % SEQ_MULTI_OF:
            raise ValueError(
                f"The control context covers {num_tokens} image tokens and the transformer's image stream is "
                f"{padded_tokens} long: the control image must be encoded at the size of the latents being denoised."
            )
        run_block = self._block_runner(inputs)
        if self.variant.refiner_chain == "plain":
            c = self._embed(control_context, x)
            for block in self.control_noise_refiner:
                c = run_block(block, c)
            return [], c
        chain = self.control_layers if self.variant.refiner_chain == "main" else self.control_noise_refiner
        return run_vace_chain(
            partial(self._embed, control_context, x),
            x,
            chain,
            run_block,
            num_hints=len(Z_IMAGE_FUN_CONTROL_REFINER_LAYERS),
        )

    def _run_main_stage(self, inputs: Mapping[str, Any], carry: list[torch.Tensor]) -> tuple[list[torch.Tensor], None]:
        unified = inputs[_STREAM]
        hints, _ = run_vace_chain(
            partial(_beside_the_caption, carry, unified),
            unified,
            self.control_layers,
            self._block_runner(inputs),
            num_hints=len(self.control_layers),
        )
        return hints, None

    def _embed(self, control_context: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """The control stream the refiner stage starts from: the context embedded, padded as the image stream ``x``."""
        num_tokens, padded_tokens = control_context.shape[1], x.shape[1]
        # Embedded at the padded length, as VideoX-Fun embeds its padded patches: what a GEMM computes for one row
        # depends on how many rows it computes at once, so this keeps the rows the reference computes.
        tokens = F.pad(control_context.to(self.dtype), (0, 0, 0, padded_tokens - num_tokens))
        c = self._embedder(tokens).expand(x.shape[0], -1, -1)
        # The image stream's padding rows are the base's `x_pad_token` here, which is what VideoX-Fun writes into the
        # control stream's padding rows. Taken from the stream, they are exactly what the base computed, however its
        # weights are stored.
        return torch.cat([c[:, :num_tokens], x[:, num_tokens:].to(c.dtype)], dim=1)

    def _block_runner(self, inputs: Mapping[str, Any]) -> BlockRunner:
        # Mask, RoPE, modulation input: the control blocks take exactly what the base block took, regional prompting's
        # attention bias included. The chain runs in the control's dtype, so the floating inputs are cast on the way in
        # and the hint on the way out (`add_hint`). RoPE is complex and passes untouched.
        passthrough = {
            name: value.to(self.dtype) if isinstance(value, torch.Tensor) and value.is_floating_point() else value
            for name, value in inputs.items()
            if name != _STREAM
        }

        def run(block: nn.Module, c: torch.Tensor) -> torch.Tensor:
            return block(c, **passthrough)

        return run


def _beside_the_caption(carry: list[torch.Tensor], unified: torch.Tensor) -> torch.Tensor:
    """The control stream the main stage starts from: the refiner stage's, popped from ``carry``, beside the refined
    caption. The unified stream is ``[image | caption]``, its image part as long as the control stream."""
    carried = carry.pop()
    return torch.cat([carried, unified[:, carried.shape[1] :].to(carried.dtype)], dim=1)


def _variant_of(sd: Mapping[str, torch.Tensor], in_features: int) -> ZImageFunControlVariant:
    num_blocks = 1 + max(
        (int(parts[1]) for key in sd if (parts := key.split("."))[0] == "control_layers" and parts[1].isdigit()),
        default=-1,
    )
    projections = [sd[key] for key in _REFINER_PROJECTIONS if key in sd]
    if not projections:
        refiner = "absent"
    elif len(projections) < len(_REFINER_PROJECTIONS):
        refiner = "incomplete"
    else:
        refiner = "trained" if any(bool(tensor.any()) for tensor in projections) else "untrained"
    variant = _LAYOUTS.get((in_features, num_blocks, refiner))
    if variant is None:
        known = "; ".join(
            f"{layout.name}: {features} features, {blocks} blocks, {projection}"
            for (features, blocks, projection), layout in _LAYOUTS.items()
        )
        raise ValueError(
            f"Not a Z-Image Fun-ControlNet layout Invoke knows: `control_all_x_embedder` takes {in_features} "
            f"features, with {num_blocks} control blocks and {refiner} refiner projections ({known})."
        )
    return variant
