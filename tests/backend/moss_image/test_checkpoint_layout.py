"""Unfusing MOSS-Image's `proj.qkv` into the `proj.q/k/v` the vendored transformer has."""

import gguf
import numpy as np
import pytest
import torch

from invokeai.backend.moss_image.checkpoint_layout import drop_comfy_attention_config, split_fused_qkv
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from tests.backend.model_manager.load.moss_image_fixtures import forward, fuse_qkv, tiny_reference, with_state_dict
from tests.fixtures.quantized_payloads import random_q4_k


def test_a_fused_bf16_checkpoint_is_the_split_one_again() -> None:
    reference = tiny_reference()
    split = {key: value.clone() for key, value in reference.state_dict().items()}

    restored = split_fused_qkv(fuse_qkv(split))

    assert restored.keys() == split.keys()
    for key, value in split.items():
        assert torch.equal(restored[key], value), key
    # The same function, not just the same tensors.
    torch.testing.assert_close(forward(with_state_dict(restored)), forward(reference))


def test_a_state_dict_without_fused_tensors_is_left_alone() -> None:
    sd = {key: value.clone() for key, value in tiny_reference().state_dict().items()}
    before = dict(sd)

    assert split_fused_qkv(sd) is sd
    assert all(sd[key] is before[key] for key in before)


def test_the_text_refiners_own_fused_qkv_is_not_split() -> None:
    sd = {"text_embedder.blocks.0.qkv.weight": torch.zeros(12, 4), "layers.0.proj.qkv.weight": torch.zeros(6, 2)}

    split_fused_qkv(sd)

    assert set(sd) == {
        "text_embedder.blocks.0.qkv.weight",
        "layers.0.proj.q.weight",
        "layers.0.proj.k.weight",
        "layers.0.proj.v.weight",
    }


def test_int8_scales_split_with_their_rows_and_the_marker_is_shared() -> None:
    codes = torch.arange(9 * 4, dtype=torch.int8).reshape(9, 4)
    scale = torch.arange(1, 10, dtype=torch.float32).reshape(9, 1)
    marker = torch.tensor([1, 2, 3], dtype=torch.uint8)
    sd = {
        "layers.2.proj.qkv.weight": codes,
        "layers.2.proj.qkv.weight_scale": scale,
        "layers.2.proj.qkv.comfy_quant": marker,
    }

    split_fused_qkv(sd)

    for index, name in enumerate("qkv"):
        rows = slice(3 * index, 3 * index + 3)
        assert torch.equal(sd[f"layers.2.proj.{name}.weight"], codes[rows])
        assert torch.equal(sd[f"layers.2.proj.{name}.weight_scale"], scale[rows])
        assert torch.equal(sd[f"layers.2.proj.{name}.comfy_quant"], marker)
    assert not any(".qkv." in key and key.startswith("layers.") for key in sd)


def test_a_per_tensor_scale_is_shared_not_cut() -> None:
    sd = {
        "layers.0.proj.qkv.weight": torch.zeros(6, 2, dtype=torch.int8),
        "layers.0.proj.qkv.weight_scale": torch.ones(()),
    }

    split_fused_qkv(sd)

    assert all(sd[f"layers.0.proj.{name}.weight_scale"].shape == () for name in "qkv")


def test_a_one_element_scale_is_shared_not_cut() -> None:
    sd = {
        "layers.0.proj.qkv.weight": torch.zeros(6, 2, dtype=torch.int8),
        "layers.0.proj.qkv.weight_scale": torch.ones(1),
    }

    split_fused_qkv(sd)

    assert all(sd[f"layers.0.proj.{name}.weight_scale"].shape == (1,) for name in "qkv")


def test_dense_and_int8_parts_own_their_storage() -> None:
    """A view would keep the whole fused tensor alive for as long as any one projection lives.

    Ownership is read off the storage size, not its address: a view reports the whole fused storage, and the allocator
    may reuse the fused tensor's address for a part once the split has freed it."""
    sd = {
        "layers.0.proj.qkv.weight": torch.zeros(6, 4, dtype=torch.int8),
        "layers.0.proj.qkv.weight_scale": torch.ones(6, 1),
        "layers.1.proj.qkv.weight": torch.zeros(6, 4, dtype=torch.bfloat16),
    }

    split_fused_qkv(sd)

    parts = [value for key, value in sd.items() if key.split(".")[3] in "qkv"]
    assert len(parts) == 9
    for part in parts:
        assert part.untyped_storage().nbytes() == part.nbytes


@pytest.mark.parametrize(
    ("qtype", "columns"),
    [(gguf.GGMLQuantizationType.Q8_0, 64), (gguf.GGMLQuantizationType.Q4_0, 64), (gguf.GGMLQuantizationType.Q4_K, 256)],
    ids=["Q8_0", "Q4_0", "Q4_K"],
)
def test_ggml_rows_are_cut_packed_and_dequantize_to_the_fused_tensors_rows(
    qtype: gguf.GGMLQuantizationType, columns: int
) -> None:
    """Cut on the packed bytes: no dequantization, yet each part decodes to its third of the fused tensor's values."""
    rows = 12
    rng = np.random.default_rng(0)
    if qtype is gguf.GGMLQuantizationType.Q4_K:
        packed = random_q4_k((rows, columns), rng)
    else:
        packed = gguf.quants.quantize(rng.standard_normal((rows, columns)).astype(np.float32), qtype)
    fused = GGMLTensor(torch.from_numpy(packed), qtype, torch.Size([rows, columns]), torch.float32)
    expected = fused.get_dequantized_tensor()
    sd = {"layers.0.proj.qkv.weight": fused}

    split_fused_qkv(sd)

    for index, name in enumerate("qkv"):
        part = sd[f"layers.0.proj.{name}.weight"]
        assert isinstance(part, GGMLTensor)
        assert part._ggml_quantization_type is qtype
        assert part.tensor_shape == torch.Size([4, columns])
        assert part.quantized_data.shape[0] == 4
        assert torch.equal(part.get_dequantized_tensor(), expected[4 * index : 4 * index + 4])


def test_parts_of_a_ggml_tensor_own_their_storage() -> None:
    """A view would keep the whole fused tensor alive for as long as any one projection lives."""
    packed = gguf.quants.quantize(
        np.random.default_rng(0).standard_normal((6, 32)).astype(np.float32), gguf.GGMLQuantizationType.Q8_0
    )
    sd = {
        "layers.0.proj.qkv.weight": GGMLTensor(
            torch.from_numpy(packed), gguf.GGMLQuantizationType.Q8_0, torch.Size([6, 32]), torch.float32
        )
    }

    split_fused_qkv(sd)

    storages = {sd[f"layers.0.proj.{name}.weight"].quantized_data.untyped_storage().data_ptr() for name in "qkv"}
    assert len(storages) == 3


@pytest.mark.parametrize(
    ("sd", "message"),
    [
        ({"layers.0.proj.qkv.weight": torch.zeros(7, 2)}, "do not split into q, k and v"),
        ({"layers.0.proj.qkv.weight_scale": torch.ones(6, 1)}, "no fused weight"),
        (
            {"layers.0.proj.qkv.weight": torch.zeros(6, 2), "layers.0.proj.k.weight": torch.zeros(2, 2)},
            "also holds layers.0.proj.k.weight",
        ),
    ],
    ids=["rows_not_divisible_by_three", "sidecar_without_weight", "fused_and_split_both_present"],
)
def test_a_malformed_fusion_is_refused(sd: dict[str, torch.Tensor], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        split_fused_qkv(sd)


def test_comfys_attention_kernel_records_are_dropped_and_nothing_else() -> None:
    sd = {
        "layers.0.comfy_attention.config": torch.zeros(3, dtype=torch.uint8),
        "layers.1.comfy_attention.config": torch.zeros(3, dtype=torch.uint8),
        "layers.0.norm1.weight": torch.ones(2),
    }

    drop_comfy_attention_config(sd)

    assert set(sd) == {"layers.0.norm1.weight"}
