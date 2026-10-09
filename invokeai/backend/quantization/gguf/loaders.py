import gc
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import gguf
import numpy as np
import torch

from invokeai.backend.quantization.fp8_scaled import COMFY_QUANT_SUFFIX
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.quantization.gguf.utils import TORCH_COMPATIBLE_QTYPES
from invokeai.backend.quantization.int8_convrot import INT8_TENSORWISE_FORMAT
from invokeai.backend.util.logging import InvokeAILogger

logger = InvokeAILogger.get_logger()


class WrappedGGUFReader:
    """Wrapper around GGUFReader that adds a close() method."""

    def __init__(self, path: Path):
        self.reader = gguf.GGUFReader(path)

    def __enter__(self):
        return self.reader

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        """Explicitly close the memory-mapped file."""
        if hasattr(self.reader, "data"):
            try:
                self.reader.data.flush()
                del self.reader.data
            except (AttributeError, OSError, ValueError) as e:
                logger.warning(f"Wasn't able to close GGUF memory map: {e}")
        del self.reader
        gc.collect()


ORIG_SHAPE_KEY_PREFIX = "comfy.gguf.orig_shape."


def _coerce_dim(value: Any) -> int | None:
    """Coerce a single value from a GGUF metadata array to a positive dimension, or None if it isn't one.

    Metadata is untrusted input, so anything that is not a finite, integral, positive number is rejected rather
    than silently truncated (``int(2.5) == 2``) or allowed to raise (``int(float("inf"))``).
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        dim = int(value)
    elif isinstance(value, (float, np.floating)):
        value = float(value)
        if not math.isfinite(value) or not value.is_integer():
            return None
        dim = int(value)
    else:
        return None
    return dim if dim > 0 else None


def _read_comfy_orig_shapes(reader: gguf.GGUFReader) -> dict[str, torch.Size]:
    """Read ComfyUI's ``comfy.gguf.orig_shape.<tensor name>`` metadata.

    ComfyUI's GGUF converter can only quantize 2-D tensors, so it reshapes any tensor whose native
    rank/shape the quantizer rejects (e.g. Krea-2's ``first.weight`` of (6144, 64)) into a workable
    2-D shape and records the native shape under this key. Without honoring it, the tensor loads with
    the reshaped shape and ``load_state_dict`` fails with a size mismatch.
    """
    orig_shapes: dict[str, torch.Size] = {}
    for key, field in reader.fields.items():
        if not key.startswith(ORIG_SHAPE_KEY_PREFIX):
            continue
        tensor_name = key[len(ORIG_SHAPE_KEY_PREFIX) :]
        try:
            contents = field.contents()
        except Exception as e:
            logger.warning(f"Ignoring malformed GGUF metadata key {key!r}: {e}")
            continue
        if not isinstance(contents, (list, tuple)):
            logger.warning(f"Ignoring malformed GGUF metadata key {key!r}: expected an array, got {contents!r}")
            continue
        dims = tuple(_coerce_dim(v) for v in contents)
        if not dims or any(d is None for d in dims):
            logger.warning(f"Ignoring malformed GGUF metadata key {key!r}: {tuple(contents)!r}")
            continue
        orig_shapes[tensor_name] = torch.Size(dims)
    return orig_shapes


COMFY_QUANT_KEY_PREFIX = "comfy.gguf.quant."

Q8CRMode = Literal["refuse", "decode", "ignore"]


def parse_q8_cr_markers(fields: Mapping[str, Any], source: str) -> dict[str, dict[str, Any]]:
    """Parse ComfyUI-GGUF's ``Q8_CR`` layers from a GGUF's ``comfy.gguf.quant.<weight name>`` metadata.

    ``Q8_CR`` (github.com/molbal/ComfyUI-GGUF) is ComfyUI's ``int8_tensorwise`` + convrot scheme in a
    GGUF container: the int8 codes are stored under GGML type ``I8`` (GGUF has no other int8 type), a
    per-output-row F32 scale beside them as ``<weight name>_scale``, and the JSON that a safetensors
    export keeps in ``<layer>.comfy_quant`` here lives in the metadata, as a string. ``I8`` is not a
    GGML quantization -- ``gguf`` has no dequantizer for it -- so these layers cannot ride the
    GGMLTensor path at all.

    ``fields`` maps metadata keys to their string values; other keys are ignored. Returns the int8
    markers keyed by weight name, in the form :mod:`int8_convrot` reads. Raises ``ValueError`` for
    anything else the converter writes (``int4_cr``, the retired ``int4_pytorch``): those tensors are
    ``I8`` too, and nothing here decodes them. Model identification turns that into the installer's
    refusal, so a file is refused for the same reason at install and at load.
    """
    layers: dict[str, dict[str, Any]] = {}
    for key, value in fields.items():
        if not key.startswith(COMFY_QUANT_KEY_PREFIX):
            continue
        weight_name = key[len(COMFY_QUANT_KEY_PREFIX) :]
        try:
            config = json.loads(value)
        except Exception as e:
            raise ValueError(f"{source}: GGUF metadata {key!r} is not valid quantization JSON: {e}") from None
        if not isinstance(config, dict) or config.get("format") != INT8_TENSORWISE_FORMAT:
            fmt = config.get("format") if isinstance(config, dict) else config
            raise ValueError(
                f"{source}: tensor {weight_name!r} uses ComfyUI-GGUF quantization {fmt!r}, which is not "
                f"supported. Only {INT8_TENSORWISE_FORMAT!r} (Q8_CR) layers can be loaded from GGUF."
            )
        if not weight_name.endswith(".weight"):
            raise ValueError(f"{source}: {INT8_TENSORWISE_FORMAT} metadata names {weight_name!r}, not a weight.")
        marker = dict(config)
        # The converter's own loader treats `convrot` without `weight_rotated` as a file from an older
        # converter that stored the codes unrotated, and runs it without the rotation. Un-rotating
        # weights that were never rotated would scramble them, so follow it.
        if marker.get("convrot") and not marker.get("weight_rotated", False):
            marker["convrot"] = False
            marker.pop("convrot_groupsize", None)
        layers[weight_name] = marker
    return layers


def gguf_sd_loader(path: Path, compute_dtype: torch.dtype, *, q8_cr: Q8CRMode = "refuse") -> dict[str, torch.Tensor]:
    """Read a GGUF file into a state dict of ``GGMLTensor``s, dequantized on the fly at use.

    ``q8_cr`` says what to do with ComfyUI-GGUF ``Q8_CR`` layers (see :func:`parse_q8_cr_markers`):

    - ``"decode"``: their weight, scale and bias come back as plain tensors with a ``<layer>.comfy_quant``
      marker beside them -- exactly a safetensors ``int8_tensorwise`` export, so the loader decodes both
      containers with the same :mod:`int8_convrot` code. Only a loader that does so may ask for this.
    - ``"refuse"``: any other loader. Raises before the tensors are read; without it the int8 codes reach
      a plain ``nn.Linear`` and the load fails on a dtype error that names neither the file nor the format.
      Identification refuses such files at install already, so this guards records made before it did.
    - ``"ignore"``: model identification, which needs shapes and never builds a module. The file reads as
      it always has; whether its quantization is supported is the installer's question, answered from
      the metadata (see ``ModelConfigFactory``), so it is not raised here once per candidate config.
    """
    with WrappedGGUFReader(path) as reader:
        sd: dict[str, torch.Tensor] = {}
        orig_shapes = _read_comfy_orig_shapes(reader)
        int8_layers: dict[str, dict[str, Any]] = {}
        if q8_cr != "ignore":
            quant_fields = {k: f.contents() for k, f in reader.fields.items() if k.startswith(COMFY_QUANT_KEY_PREFIX)}
            int8_layers = parse_q8_cr_markers(quant_fields, path.name)
        if int8_layers and q8_cr == "refuse":
            raise ValueError(
                f"{path.name} holds {len(int8_layers)} ComfyUI-GGUF Q8_CR (int8 convrot) layer(s), e.g. "
                f"{next(iter(int8_layers))!r}. Loading Q8_CR from GGUF is not supported for this model type; "
                "use a GGML-quantized (Q8_0, Q4_K, ...) or safetensors build instead."
            )
        int8_scales = {f"{name}_scale" for name in int8_layers}
        int8_biases = {f"{name[: -len('.weight')]}.bias" for name in int8_layers}
        missing = sorted((int8_layers.keys() | int8_scales) - {tensor.name for tensor in reader.tensors})
        if missing:
            raise ValueError(f"{path.name}: Q8_CR metadata names layers the file does not hold, e.g. {missing[0]!r}.")

        for tensor in reader.tensors:
            # Use .copy() to create a true copy of the data, not a view.
            # This is critical on Windows where the memory-mapped file cannot be deleted
            # while tensors still hold references to the mapped memory.
            torch_tensor = torch.from_numpy(tensor.data.copy())

            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
            orig_shape = orig_shapes.get(tensor.name)
            if orig_shape is not None:
                if orig_shape.numel() != shape.numel():
                    raise ValueError(
                        f"GGUF tensor {tensor.name!r} declares original shape {tuple(orig_shape)}, which has a "
                        f"different element count than its stored shape {tuple(shape)}."
                    )
                shape = orig_shape

            if tensor.name in int8_layers or tensor.name in int8_scales:
                expected = gguf.GGMLQuantizationType.I8 if tensor.name in int8_layers else gguf.GGMLQuantizationType.F32
                if tensor.tensor_type != expected:
                    raise ValueError(
                        f"{path.name}: Q8_CR tensor {tensor.name!r} is stored as {tensor.tensor_type.name}, "
                        f"expected {expected.name}."
                    )
                sd[tensor.name] = torch_tensor.reshape(shape)
                continue

            if tensor.tensor_type in TORCH_COMPATIBLE_QTYPES:
                torch_tensor = torch_tensor.view(*shape)
            ggml_tensor = GGMLTensor(
                torch_tensor,
                ggml_quantization_type=tensor.tensor_type,
                tensor_shape=shape,
                compute_dtype=compute_dtype,
            )
            # An int8 layer's bias becomes a buffer of `Int8ConvrotLinear`, whose forward casts it to the
            # activation dtype -- a dtype change a GGMLTensor refuses.
            sd[tensor.name] = ggml_tensor.get_dequantized_tensor() if tensor.name in int8_biases else ggml_tensor

        for weight_name, marker in int8_layers.items():
            raw = json.dumps(marker).encode("utf-8")
            marker_key = f"{weight_name[: -len('.weight')]}{COMFY_QUANT_SUFFIX}"
            sd[marker_key] = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        return sd
