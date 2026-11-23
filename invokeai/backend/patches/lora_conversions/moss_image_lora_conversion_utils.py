"""MOSS-Image LoRAs onto `MOSSImageTransformer`; which keys a file may hold is `moss_image_lora_constants`."""

from collections.abc import Mapping

import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.merged_layer_patch import MergedLayerPatch, Range
from invokeai.backend.patches.layers.utils import any_lora_layer_from_state_dict
from invokeai.backend.patches.lora_conversions.moss_image_lora_constants import (
    MOSS_IMAGE_LORA_TRANSFORMER_PREFIX,
    describe_refusal,
    is_fused_qkv,
    read_moss_image_lora_keys,
    split_qkv_modules,
)
from invokeai.backend.patches.lora_conversions.peft_adapter_utils import peft_scale
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.util.logging import InvokeAILogger

logger = InvokeAILogger.get_logger(__name__)

Values = dict[str, torch.Tensor]

# The values of a fused layer that have one row per output row, and so are cut with them.
_ROW_KEYS = ("diff", "lora_up.weight", "dora_scale", "dora_magnitude", "hada_w1_a", "hada_w2_a")


def lora_model_from_moss_image_state_dict(
    state_dict: dict[str, torch.Tensor], *, name: str, metadata: Mapping[str, str] | None = None
) -> ModelPatchRaw:
    """Convert the MOSS-Image LoRA `name`, in any of the layouts `moss_image_lora_constants` reads, to a patch for the
    transformer, keyed `MOSS_IMAGE_LORA_TRANSFORMER_PREFIX` + module path. `metadata` is the file's safetensors header.

    A layer's `alpha` in the file sets its scale. A low-rank layer without one takes the scale PEFT gives it where the
    header holds diffusers' PEFT config (`peft_scale`), and 1 otherwise (its alpha is its rank), as ComfyUI applies it.
    A block's fused `proj.qkv` becomes the `proj.q`, `proj.k` and `proj.v` the transformer has, exactly
    (`_split_fused_qkv`). Shapes are not checked here (see `moss_image_lora_constants`).
    """
    keys = read_moss_image_lora_keys(state_dict)
    if keys.refused:
        # Identification refuses these files, so one reaches here only by another path; applying the rest would make a
        # partial LoRA that passes for a whole one.
        raise ValueError(f"LoRA '{name}': {describe_refusal(keys.refused)}")
    scale = peft_scale(metadata, family="MOSS-Image")

    values_by_module: dict[str, Values] = {
        module: {value_key: state_dict[key] for value_key, key in sources.items()}
        for module, sources in keys.layers.items()
    }
    leftovers = {
        module: {value_key: state_dict[key] for value_key, key in sources.items()}
        for module, sources in keys.leftovers.items()
    }
    for fused in [module for module in values_by_module if is_fused_qkv(module)] + [
        module for module in leftovers if is_fused_qkv(module)
    ]:
        _complete_leftovers(fused, values_by_module, leftovers)

    layers: dict[str, BaseLayerPatch] = {}
    for module, values in values_by_module.items():
        if scale is not None and "alpha" not in values and "lora_down.weight" in values:
            # The layer classes scale by alpha / rank.
            values["alpha"] = torch.tensor(scale * values["lora_down.weight"].shape[0])
        if is_fused_qkv(module):
            built = dict(zip(split_qkv_modules(module), _split_fused_qkv(values), strict=True))
        else:
            built = {module: any_lora_layer_from_state_dict(values)}
        for path, layer in built.items():
            layers[f"{MOSS_IMAGE_LORA_TRANSFORMER_PREFIX}{path}"] = layer
    return ModelPatchRaw(layers=layers)


def _complete_leftovers(fused: str, values_by_module: dict[str, Values], leftovers: dict[str, Values]) -> None:
    """Resolve the two incomplete entries ai-toolkit writes beside a fused `proj.qkv` (`MossImageLoRAKeys.leftovers`).

    A lone `dora_scale` there is the output-axis magnitude of the DoRA layers its split LoRA make: each takes its third.
    The lone `alpha` of `proj.q`/`k`/`v` is the LoKr's own: ai-toolkit saves one `proj.qkv.alpha` cloned onto q, k and
    v (its `convert_lora_weights_before_save` lists `alpha` among the shared keys) and keeps the LoKr keys fused.
    The LoKr takes it, which is the scale it trained with: ai-toolkit scales a LoKr by alpha / rank where an operand
    is factored, and by 1 where both are whole. ComfyUI never reads these alphas for the fused LoKr and applies 1.
    Alphas that differ do not come from ai-toolkit, and identification, which sees shapes only, cannot compare
    them: the first, in q, k, v order, is taken and the conversion warns.
    """
    modules = split_qkv_modules(fused)
    if fused in leftovers:
        for module, third in zip(modules, leftovers[fused]["dora_scale"].chunk(3, dim=0), strict=True):
            values_by_module[module]["dora_scale"] = third
        return
    alphas = [leftovers[module]["alpha"] for module in modules if module in leftovers]
    if alphas:
        if any(not torch.equal(alpha, alphas[0]) for alpha in alphas):
            logger.warning(
                f"The lone alphas beside {fused} differ ({[alpha.item() for alpha in alphas]}); its LoKr takes the "
                f"first, {alphas[0].item()}."
            )
        values_by_module[fused]["alpha"] = alphas[0]


def _operand_rows(values: Values, operand: str) -> int:
    """The rows of a LoKr operand (`lokr_w1` or `lokr_w2`), whole or as the product of its factors."""
    return (values[operand] if operand in values else values[f"{operand}_a"]).shape[0]


def _split_fused_qkv(values: Values) -> list[BaseLayerPatch]:
    """The q, k and v layers a fused `proj.qkv` layer applies as. Exact for every kind, as a row of the update depends
    on the same row of the factors alone:

    - low-rank, DoRA with an output-axis magnitude, and a full diff: the rows of `lora_up` (and of the magnitude and
      the diff) are cut in thirds. `lora_down` and `alpha` are shared, which holds for a block-diagonal `lora_up` of
      rank 3r too, and the layers keep the low-rank sidecar path;
    - LoHA: (A1 B1)[rows] * (A2 B2)[rows] = (A1[rows] B1) * (A2[rows] B2), so the `_a` factors are cut;
    - LoKr: `_lokr_rows`.
    """
    if "lokr_w1" in values or "lokr_w1_a" in values:
        third = _operand_rows(values, "lokr_w1") * _operand_rows(values, "lokr_w2") // 3
        return [_lokr_rows(values, i * third, (i + 1) * third) for i in range(3)]
    row_keys = [key for key in _ROW_KEYS if key in values]
    third = values[row_keys[0]].shape[0] // 3
    return [
        any_lora_layer_from_state_dict(
            {**values, **{key: values[key][i * third : (i + 1) * third] for key in row_keys}}
        )
        for i in range(3)
    ]


def _slice_operand(values: Values, operand: str, start: int, stop: int) -> Values:
    """`values` with rows `start`..`stop` of a LoKr operand: of the operand itself, or of the first factor of its
    product, whose rows are the product's."""
    key = operand if operand in values else f"{operand}_a"
    return {**values, key: values[key][start:stop]}


def _lokr_rows(values: Values, start: int, stop: int) -> BaseLayerPatch:
    """The LoKr layer for rows `start`..`stop` of a LoKr's update, kron(w1, w2): row i * c + j of it is
    kron(w1[i], w2[j]), c being the rows of w2. The span is cut into LoKr pieces, each exact and materializing nothing
    the layer would not: kron(w1[i0:i1], w2) for the blocks of w2 rows it holds whole, and kron(w1[i:i+1], w2[j0:j1])
    for a block it cuts through. A span of more than one piece is a `MergedLayerPatch` of them.

    Every piece holds its own operands (the whole `w2` among them, for each piece that keeps it whole). Direct
    patching merges a piece and frees it, but sidecars on GGUF, int8 and FP8 Storage builds are moved to the device
    together, so a fused qkv LoKr's `w2` lies there several times over (the reviewers measured about 4x at factor 4
    and 5x at factors 8 and 16). The reservation prices it through `_elements`."""
    c = _operand_rows(values, "lokr_w2")
    pieces: list[BaseLayerPatch] = []
    ranges: list[Range] = []
    row = start
    while row < stop:
        i, j = divmod(row, c)
        if j == 0 and stop - row >= c:
            blocks = (stop - row) // c
            piece = _slice_operand(values, "lokr_w1", i, i + blocks)
            span = blocks * c
        else:
            span = min(c - j, stop - row)
            piece = _slice_operand(_slice_operand(values, "lokr_w1", i, i + 1), "lokr_w2", j, j + span)
        pieces.append(any_lora_layer_from_state_dict(piece))
        ranges.append(Range(row - start, row - start + span))
        row += span
    return pieces[0] if len(pieces) == 1 else MergedLayerPatch(pieces, ranges)
