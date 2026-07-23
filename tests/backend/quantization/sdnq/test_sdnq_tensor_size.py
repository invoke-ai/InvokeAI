"""Tests that cache byte accounting reflects an SDNQTensor's real storage.

calc_tensor_size() must count the packed uint4/int5 data plus every auxiliary payload (scale,
zero_point, svd), not the wrapper's advertised dequantized shape with a uint8 dtype (which
over-counts packed weights and omits the auxiliary tensors).
"""

import torch

from invokeai.backend.quantization.sdnq.sdnq_tensor import SDNQTensor
from invokeai.backend.quantization.sdnq.utils import SDNQQuantizationType
from invokeai.backend.util.calc_tensor_size import calc_tensor_size


def _nbytes(*tensors: torch.Tensor) -> int:
    return sum(t.nelement() * t.element_size() for t in tensors)


def test_uint4_with_svd_size_is_packed_plus_auxiliary():
    out_features, in_features, num_groups = 4, 8, 2
    data = torch.randint(0, 256, (out_features, in_features // 2), dtype=torch.uint8)  # packed uint4
    scale = torch.rand(out_features, num_groups, 1, dtype=torch.float32)
    zero_point = torch.rand(out_features, num_groups, 1, dtype=torch.float32)
    svd_up = torch.rand(out_features, 2, dtype=torch.float32)
    svd_down = torch.rand(2, in_features, dtype=torch.float32)

    t = SDNQTensor(
        data=data,
        quantization_type=SDNQQuantizationType.UINT4_ASYM,
        tensor_shape=torch.Size([out_features, in_features]),
        compute_dtype=torch.bfloat16,
        scale=scale,
        zero_point=zero_point,
        svd_up=svd_up,
        svd_down=svd_down,
        group_size=4,
    )

    expected = _nbytes(data, scale, zero_point, svd_up, svd_down)
    assert t.sdnq_storage_nbytes() == expected
    assert calc_tensor_size(t) == expected
    # The naive accounting (dequantized shape × uint8) both over-counts and omits the aux payloads.
    assert calc_tensor_size(t) != t.nelement() * t.element_size()


def test_int5_size_is_packed_plus_scale():
    out_features, in_features, num_groups = 2, 8, 2
    data = torch.randint(0, 256, (out_features, 5), dtype=torch.uint8)  # 8 int5 values per 5 bytes
    scale = torch.rand(out_features, num_groups, 1, dtype=torch.float32)

    t = SDNQTensor(
        data=data,
        quantization_type=SDNQQuantizationType.INT5_ASYM,
        tensor_shape=torch.Size([out_features, in_features]),
        compute_dtype=torch.bfloat16,
        scale=scale,
        zero_point=None,
        group_size=4,
    )

    expected = _nbytes(data, scale)
    assert t.sdnq_storage_nbytes() == expected
    assert calc_tensor_size(t) == expected
