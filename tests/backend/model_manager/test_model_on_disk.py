"""The header readers identification and loaders use instead of reading a model's tensors."""

import json
from pathlib import Path

import gguf
import pytest

from invokeai.backend.model_manager.model_on_disk import ModelOnDisk, safetensors_meta_state_dict


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
