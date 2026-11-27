"""Tests that per-group dequantization accepts 2D scale/zero-point tensors.

SDNQ stores per-group scale/zero_point as either [out_features, num_groups, 1] or, without the
trailing singleton, [out_features, num_groups]. A 2D param must be normalized before arithmetic;
otherwise it right-aligns against the 3D grouped weight ([out, num_groups, group_size]) and fails
broadcasting.
"""

import torch

from invokeai.backend.quantization.sdnq.utils import dequantize_int5_per_group, dequantize_uint4_per_group


def test_uint4_per_group_accepts_2d_scale_and_zero_point():
    out_features, in_features, group_size = 4, 8, 4
    num_groups = in_features // group_size  # 2
    packed = torch.randint(0, 256, (out_features, in_features // 2), dtype=torch.uint8)
    scale_3d = torch.rand(out_features, num_groups, 1, dtype=torch.float32) * 0.01
    zp_3d = torch.rand(out_features, num_groups, 1, dtype=torch.float32) * 0.01

    expected = dequantize_uint4_per_group(
        packed, scale_3d, zp_3d, torch.Size([out_features, in_features]), group_size, dtype=torch.float32
    )
    got = dequantize_uint4_per_group(
        packed,
        scale_3d.squeeze(-1),
        zp_3d.squeeze(-1),
        torch.Size([out_features, in_features]),
        group_size,
        dtype=torch.float32,
    )

    assert got.shape == torch.Size([out_features, in_features])
    assert torch.equal(got, expected)


def test_int5_per_group_accepts_2d_scale_and_zero_point():
    out_features, in_features, group_size = 2, 8, 4
    num_groups = in_features // group_size  # 2
    # int5 packs 8 values into 5 bytes: [out, 8] -> packed last dim 5.
    packed = torch.randint(0, 256, (out_features, 5), dtype=torch.uint8)
    scale_3d = torch.rand(out_features, num_groups, 1, dtype=torch.float32) * 0.01
    zp_3d = torch.rand(out_features, num_groups, 1, dtype=torch.float32) * 0.01

    expected = dequantize_int5_per_group(
        packed, scale_3d, zp_3d, torch.Size([out_features, in_features]), group_size, dtype=torch.float32
    )
    got = dequantize_int5_per_group(
        packed,
        scale_3d.squeeze(-1),
        zp_3d.squeeze(-1),
        torch.Size([out_features, in_features]),
        group_size,
        dtype=torch.float32,
    )

    assert got.shape == torch.Size([out_features, in_features])
    assert torch.equal(got, expected)


def test_int5_per_group_2d_scale_without_zero_point():
    out_features, in_features, group_size = 2, 8, 4
    num_groups = in_features // group_size
    packed = torch.randint(0, 256, (out_features, 5), dtype=torch.uint8)
    scale_2d = torch.rand(out_features, num_groups, dtype=torch.float32) * 0.01

    got = dequantize_int5_per_group(
        packed, scale_2d, None, torch.Size([out_features, in_features]), group_size, dtype=torch.float32
    )
    assert got.shape == torch.Size([out_features, in_features])
