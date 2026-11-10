"""``ModelOnDisk.metadata`` for GGUF files, which identification reads instead of reopening the file."""

from pathlib import Path

import gguf

from invokeai.backend.model_manager.model_on_disk import ModelOnDisk


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
