"""Rename a llama.cpp Qwen3-VL ``mmproj`` GGUF into the keys of the HuggingFace ``Qwen3VLVisionModel``.

llama.cpp's converter renames the tower's tensors one for one and changes only two things: the Conv3d patch
embedding is split along its temporal axis into two Conv2d weights (``v.patch_embd.weight`` is frame 0,
``v.patch_embd.weight.1`` frame 1), and each deepstack merger is named by the vision block it reads
(``v.deepstack.8``) rather than by its position in ``deepstack_merger_list``. The fused QKV and every other
tensor keep their HuggingFace layout, so the tensors need no reshaping beyond the stack.
"""

import re
from collections.abc import Mapping, Sequence

import torch

_BLOCK_TENSORS = {
    "attn_qkv": "attn.qkv",
    "attn_out": "attn.proj",
    "ffn_up": "mlp.linear_fc1",
    "ffn_down": "mlp.linear_fc2",
    "ln1": "norm1",
    "ln2": "norm2",
}
_DEEPSTACK_TENSORS = {"norm": "norm", "fc1": "linear_fc1", "fc2": "linear_fc2"}
# The ``merger`` is llama.cpp's ``mm.0`` / ``mm.2`` (the two Linears around its GELU) behind ``v.post_ln``.
_FIXED_TENSORS = {
    "v.patch_embd.bias": "patch_embed.proj.bias",
    "v.position_embd.weight": "pos_embed.weight",
    "v.post_ln.weight": "merger.norm.weight",
    "v.post_ln.bias": "merger.norm.bias",
    "mm.0.weight": "merger.linear_fc1.weight",
    "mm.0.bias": "merger.linear_fc1.bias",
    "mm.2.weight": "merger.linear_fc2.weight",
    "mm.2.bias": "merger.linear_fc2.bias",
}
_PATCH_EMBED_FRAMES = ("v.patch_embd.weight", "v.patch_embd.weight.1")
_BLOCK_KEY = re.compile(r"v\.blk\.(\d+)\.(\w+)\.(weight|bias)")
_DEEPSTACK_KEY = re.compile(r"v\.deepstack\.(\d+)\.(\w+)\.(weight|bias)")


def mmproj_to_vision_tower_state_dict(
    tensors: Mapping[str, torch.Tensor], deepstack_visual_indexes: Sequence[int]
) -> dict[str, torch.Tensor]:
    """The ``Qwen3VLVisionModel`` state dict an mmproj's plain tensors stand for.

    `deepstack_visual_indexes` is the vision config's list of the blocks whose output feeds a deepstack merger.
    A tensor this mapping has no place for raises rather than being dropped: it means the file is not the
    projector the config describes, and a tower missing part of its input would still run.
    """
    out: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        if name in _PATCH_EMBED_FRAMES:
            continue
        if name in _FIXED_TENSORS:
            out[_FIXED_TENSORS[name]] = tensor
        elif (block := _BLOCK_KEY.fullmatch(name)) and block[2] in _BLOCK_TENSORS:
            out[f"blocks.{block[1]}.{_BLOCK_TENSORS[block[2]]}.{block[3]}"] = tensor
        elif (stack := _DEEPSTACK_KEY.fullmatch(name)) and stack[2] in _DEEPSTACK_TENSORS:
            if int(stack[1]) not in deepstack_visual_indexes:
                raise ValueError(
                    f"The mmproj has a deepstack merger on block {stack[1]}, but this Qwen3-VL's deepstack blocks "
                    f"are {list(deepstack_visual_indexes)}."
                )
            position = list(deepstack_visual_indexes).index(int(stack[1]))
            out[f"deepstack_merger_list.{position}.{_DEEPSTACK_TENSORS[stack[2]]}.{stack[3]}"] = tensor
        else:
            raise ValueError(f"The mmproj has a tensor this Qwen3-VL vision tower has no place for: {name!r}.")

    missing = [name for name in _PATCH_EMBED_FRAMES if name not in tensors]
    if missing:
        raise ValueError(f"The mmproj lacks the patch embedding tensor(s) {missing}.")
    # Conv3d weight (out, in, time, height, width): frame 0 and frame 1 stacked on the time axis.
    out["patch_embed.proj.weight"] = torch.stack([tensors[name] for name in _PATCH_EMBED_FRAMES], dim=2)
    return out
