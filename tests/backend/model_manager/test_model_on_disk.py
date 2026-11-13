"""The header readers identification and loaders use instead of reading a model's tensors."""

import json
from pathlib import Path

import gguf
import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.model_manager.model_on_disk import (
    SAFETENSORS_DTYPES,
    ModelOnDisk,
    read_safetensors_header,
    safetensors_meta_state_dict,
)


def test_gguf_metadata_keeps_scalar_fields_with_their_types_and_leaves_arrays_out(tmp_path: Path) -> None:
    """The Wan UMT5 config reads ``t5encoder.block_count`` and the tokenizer's ids and flags from here;
    with strings only, every real UMT5 GGUF would be refused at install as incomplete. Arrays stay out:
    a tokenizer vocabulary is hundreds of thousands of entries, cached for the whole identification."""
    path = tmp_path / "encoder.gguf"
    writer = gguf.GGUFWriter(str(path), "t5encoder")
    writer.add_uint32("t5encoder.block_count", 24)
    writer.add_bool("tokenizer.ggml.add_eos_token", True)
    writer.add_string("tokenizer.ggml.model", "t5")
    writer.add_token_list(["<pad>", "</s>", "<unk>"])
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    metadata = ModelOnDisk(path).metadata()

    assert metadata["general.architecture"] == "t5encoder"
    assert metadata["tokenizer.ggml.model"] == "t5"
    assert metadata["t5encoder.block_count"] == 24 and type(metadata["t5encoder.block_count"]) is int
    assert metadata["tokenizer.ggml.add_eos_token"] is True
    assert "tokenizer.ggml.tokens" not in metadata


def test_a_safetensors_dtype_without_a_torch_dtype_is_refused_by_name(tmp_path: Path) -> None:
    """A sub-byte format's header counts 4-bit elements, so there is no meta tensor to build for it. The refusal has to
    name the tensor and the dtype, not surface as a bare ``KeyError`` out of a loader."""
    header = json.dumps({"layer.weight": {"dtype": "F4", "shape": [4], "data_offsets": [0, 2]}}).encode()
    path = tmp_path / "packed.safetensors"
    path.write_bytes(len(header).to_bytes(8, "little") + header + bytes(2))

    with pytest.raises(ValueError, match=r"'layer\.weight' as F4"):
        safetensors_meta_state_dict(path)


def _torch_dtypes() -> list[torch.dtype]:
    return sorted({value for value in vars(torch).values() if isinstance(value, torch.dtype)}, key=str)


@pytest.mark.parametrize("dtype", _torch_dtypes(), ids=str)
def test_every_dtype_safetensors_writes_reads_back_from_the_header(tmp_path: Path, dtype: torch.dtype) -> None:
    """The single-file loaders read through the header, so a dtype `safetensors` writes but the header table lacks
    fails a checkpoint the full `load_file` reads -- at generation, after identification has installed it through its
    fallback. Asked of safetensors itself rather than a list, so a dtype it gains is covered too. The sub-byte formats
    are the one exception, refused by name: their header shapes count elements no torch dtype stores one by one."""
    path = tmp_path / "tensor.safetensors"
    try:
        save_file({"tensor": torch.zeros(2, 3, dtype=dtype)}, str(path))
    except Exception:
        # Never for a dtype the table maps: a platform that cannot write it must not turn its coverage into a skip.
        if dtype in SAFETENSORS_DTYPES.values():
            raise
        pytest.skip(f"safetensors or torch cannot write {dtype}")

    stored_as = read_safetensors_header(path)["tensor"]["dtype"]
    if stored_as in ("F4", "F6_E2M3", "F6_E3M2"):
        with pytest.raises(ValueError, match=f"as {stored_as}"):
            safetensors_meta_state_dict(path)
        return
    meta = safetensors_meta_state_dict(path)["tensor"]

    assert meta.dtype is dtype and meta.shape == (2, 3) and meta.is_meta
