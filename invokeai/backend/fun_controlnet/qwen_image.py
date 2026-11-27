"""Qwen-Image Fun-ControlNet-Union: the control branch alibaba-pai trained for the Qwen-Image generation transformer.

Reference: VideoX-Fun ``videox_fun/models/qwenimage_transformer2d_control.py`` and
``pipeline_qwenimage_control.py``; ComfyUI mirrors it in ``comfy/ldm/qwen_image/controlnet.py``.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import accelerate
import torch
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformerBlock
from torch import nn

from invokeai.backend.fun_controlnet.branch import BlockRunner, FunControlBranch

# VideoX-Fun `config/qwenimage/qwenimage_control.yaml`: five control blocks, their hints added after these base
# blocks. The checkpoint carries no config, so the map is a constant of the family.
QWEN_IMAGE_FUN_CONTROL_LAYERS: tuple[int, ...] = (0, 12, 24, 36, 48)
# 33 latent channels -- control latents (16), inpaint keep mask (1), masked-image latents (16) -- packed 2x2.
QWEN_IMAGE_FUN_CONTROL_IN_DIM = 132
# Qwen-Image attention heads are 128 wide. The checkpoint does not say, so the head count follows from the width.
QWEN_IMAGE_ATTENTION_HEAD_DIM = 128

_IMAGE_STREAM = "hidden_states"
_TEXT_STREAM = "encoder_hidden_states"


class QwenImageFunControlBlock(QwenImageTransformerBlock):
    """A Qwen-Image block plus the two projections of the control recipe; ``forward`` is the base block's."""

    def __init__(self, *, dim: int, num_attention_heads: int, attention_head_dim: int, first: bool) -> None:
        super().__init__(dim=dim, num_attention_heads=num_attention_heads, attention_head_dim=attention_head_dim)
        if first:
            self.before_proj = nn.Linear(dim, dim)
        self.after_proj = nn.Linear(dim, dim)


class QwenImageFunControlBranch(FunControlBranch):
    """The branch for ``QwenImageTransformer2DModel``: control blocks thread their own text stream beside the base's.

    The hint is added to the image stream only. ``modulate_index`` and ``zero_cond_t`` belong to the Edit transformer,
    which the checkpoint was not trained on; the denoise refuses that pairing before anything is installed.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        control_in_dim: int = QWEN_IMAGE_FUN_CONTROL_IN_DIM,
        control_layers: Sequence[int] = QWEN_IMAGE_FUN_CONTROL_LAYERS,
    ) -> None:
        blocks = [
            QwenImageFunControlBlock(
                dim=hidden_size,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
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
        control_layers: Sequence[int] = QWEN_IMAGE_FUN_CONTROL_LAYERS,
        attention_head_dim: int = QWEN_IMAGE_ATTENTION_HEAD_DIM,
    ) -> "QwenImageFunControlBranch":
        """Build the branch the checkpoint describes and load it, cast to ``dtype``.

        The geometry is read off the tensors: the width and input features from ``control_img_in.weight``, the block
        count from the highest ``control_blocks.N`` index. Anything that is not the Qwen-Image recipe is refused by
        name rather than loaded into a branch that would run and produce noise.
        """
        weight = sd.get("control_img_in.weight")
        if weight is None or weight.dim() != 2:
            raise ValueError("Not a Qwen Image Fun-ControlNet checkpoint: `control_img_in.weight` is missing.")
        hidden_size, control_in_dim = weight.shape
        if control_in_dim != QWEN_IMAGE_FUN_CONTROL_IN_DIM:
            raise ValueError(
                f"Qwen Image Fun-ControlNet takes {QWEN_IMAGE_FUN_CONTROL_IN_DIM} control features per token (33 latent "
                f"channels packed 2x2); this checkpoint's `control_img_in` takes {control_in_dim}."
            )
        if hidden_size % attention_head_dim:
            raise ValueError(
                f"A {hidden_size}-wide branch cannot be split into {attention_head_dim}-wide attention heads."
            )
        layers = tuple(control_layers)
        block_indices = {
            int(parts[1])
            for key in sd
            if (parts := key.split(".")) and parts[0] == "control_blocks" and parts[1].isdigit()
        }
        num_blocks = max(block_indices, default=-1) + 1
        if num_blocks != len(layers):
            raise ValueError(
                f"This checkpoint has {num_blocks} control blocks; the Qwen Image recipe places {len(layers)} after base "
                f"layers {layers}."
            )

        with accelerate.init_empty_weights():
            branch = cls(
                hidden_size=hidden_size,
                num_attention_heads=hidden_size // attention_head_dim,
                attention_head_dim=attention_head_dim,
                control_in_dim=control_in_dim,
                control_layers=layers,
            )
        missing, unexpected = branch.load_state_dict(dict(sd), strict=False, assign=True)
        if missing or unexpected:
            raise ValueError(
                "This checkpoint does not match the Qwen Image control block layout: "
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
        # Mask, modulation input, RoPE, attention kwargs: the control blocks take exactly what the base block took.
        # VideoX-Fun hands the base's `encoder_hidden_states_mask` to the control blocks; ComfyUI's mirror forces it to
        # None. Both agree whenever every text token is valid, which is when Invoke's encoder passes None, so the
        # VideoX reading is kept here -- do not "fix" it toward ComfyUI.
        # The chain runs in the branch's dtype end to end: the floating inputs (the modulation input) are cast on the
        # way in, like `c` and the text stream, and the hint on the way out (`add_hint`). RoPE is complex and the mask
        # boolean; neither is touched.
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
