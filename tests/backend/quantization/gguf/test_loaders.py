from collections.abc import Sequence

import gguf
import numpy as np
import pytest
import torch

from invokeai.backend.quantization.gguf.loaders import gguf_sd_loader


def _write_gguf(path, *, orig_shape: Sequence[float] | None) -> None:
    """Write a tiny GGUF holding one F32 tensor stored 2-D as (256, 1536) i.e. torch (1536, 256)."""
    writer = gguf.GGUFWriter(str(path), "krea2")
    stored = np.arange(1536 * 256, dtype=np.float32).reshape(1536, 256)
    writer.add_tensor("first.weight", stored, raw_dtype=gguf.GGMLQuantizationType.F32)
    if orig_shape is not None:
        writer.add_array("comfy.gguf.orig_shape.first.weight", list(orig_shape))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_gguf_sd_loader_honors_comfy_orig_shape(tmp_path):
    """ComfyUI reshapes non-2-D tensors before quantizing; the recorded native shape must win."""
    path = tmp_path / "model.gguf"
    _write_gguf(path, orig_shape=(6144, 64))

    sd = gguf_sd_loader(path, compute_dtype=torch.bfloat16)

    assert tuple(sd["first.weight"].shape) == (6144, 64)
    assert tuple(sd["first.weight"].get_dequantized_tensor().shape) == (6144, 64)


def test_gguf_sd_loader_without_orig_shape(tmp_path):
    path = tmp_path / "model.gguf"
    _write_gguf(path, orig_shape=None)

    sd = gguf_sd_loader(path, compute_dtype=torch.bfloat16)

    assert tuple(sd["first.weight"].shape) == (1536, 256)


def test_gguf_sd_loader_rejects_orig_shape_with_wrong_element_count(tmp_path):
    path = tmp_path / "model.gguf"
    _write_gguf(path, orig_shape=(6144, 65))

    with pytest.raises(ValueError, match="different element count"):
        gguf_sd_loader(path, compute_dtype=torch.bfloat16)


@pytest.mark.parametrize(
    "orig_shape",
    [
        pytest.param([2.5, 157286.4], id="non-integral"),
        pytest.param([float("inf")], id="infinite"),
        pytest.param([float("nan"), 64.0], id="nan"),
        pytest.param([-6144.0, -64.0], id="negative"),
    ],
)
def test_gguf_sd_loader_ignores_malformed_orig_shape(tmp_path, orig_shape):
    """Malformed metadata must be ignored with a warning - never truncated to a wrong shape, never raised."""
    path = tmp_path / "model.gguf"
    _write_gguf(path, orig_shape=orig_shape)

    sd = gguf_sd_loader(path, compute_dtype=torch.bfloat16)

    assert tuple(sd["first.weight"].shape) == (1536, 256)
