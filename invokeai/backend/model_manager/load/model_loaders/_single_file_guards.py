"""Refusals a single-file loader applies before it spends anything on a checkpoint it cannot read.

Shared by the loaders that accept ComfyUI's quantized single-file releases (MiniMax H3, LTX-2, ...):
each names the formats it can consume and gets the same two gates, worded for its own files.
"""

from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any


def reject_formats_declared_in_the_header(
    model_path: Path, what: str, logger: Any, accepted_formats: Collection[str], supported_note: str
) -> None:
    """Refuse a quantization format the caller cannot read, from the header alone.

    Before the tensor read, which is ~20 GiB for a transformer. Both header transports are read,
    because which one a file uses depends on the tool that produced it, not on the format: a
    per-tensor ``.comfy_quant`` marker, or an entry in ``_quantization_metadata``. FLUX.2's Comfy-Org
    fp8 build uses only the second, and reading only the first let a file of that shape through to a
    converter that died with a bare ``IndexError`` naming neither fp8 nor the format.

    The two are read with different strictness on purpose. A per-tensor marker exists only to declare
    a scheme, so one without a readable ``format`` is refused as unreadable. A header entry can carry
    per-layer flags and nothing else -- ``full_precision_matrix_mult`` alone is a well-formed entry --
    so one without a ``format`` declares nothing and is skipped; refusing it would turn away a valid
    build for a hint this loader does not even read. An entry that is not a mapping at all is refused
    as unreadable rather than left to raise ``AttributeError``.
    """
    from invokeai.backend.quantization.fp8_scaled import parse_quantization_metadata, read_safetensors_metadata
    from invokeai.backend.quantization.int8_convrot import read_comfy_quant_markers

    declared = [marker.get("format") for marker in read_comfy_quant_markers(model_path).values()]
    for entry in parse_quantization_metadata(read_safetensors_metadata(model_path, logger)).values():
        if not isinstance(entry, Mapping):
            declared.append(None)
        elif "format" in entry:
            declared.append(entry["format"])
    unsupported = sorted({str(fmt or "unreadable") for fmt in declared if fmt not in accepted_formats})
    if unsupported:
        raise ValueError(
            f"Unsupported quantization format(s) {unsupported} in {what} {model_path.name}. {supported_note}"
        )


def reject_float8_weights(sd: Mapping[str, Any], what: str, model_path: Path, supported_note: str) -> None:
    """Refuse float8 weights that nothing in the header declared.

    The older ComfyUI scaled-fp8 shape declares itself nowhere -- fp8 codes and a ``weight_scale``
    beside them, no marker, no header entry -- so the header gate cannot see it. Refused here, after
    the read but before any converter, which is where it otherwise fails: in a qkv split on a
    per-tensor scale, or at ``load_state_dict`` on an orphaned scale reported as an "unexpected key".
    Keyed on the dtype rather than on a scale key, so a raw fp8 file with no scale is named too
    instead of loading fp8 parameters that fail at the first matmul.
    """
    from invokeai.backend.quantization.fp8_scaled import FP8_WEIGHT_DTYPES

    carried = sorted(key for key, value in sd.items() if getattr(value, "dtype", None) in FP8_WEIGHT_DTYPES)
    if carried:
        raise ValueError(
            f"{what} {model_path.name} carries {len(carried)} float8 weight(s) (e.g. {', '.join(carried[:3])}). "
            f"{supported_note}"
        )
