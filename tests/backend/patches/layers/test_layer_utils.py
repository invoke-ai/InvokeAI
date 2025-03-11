import torch

from invokeai.backend.patches.layers.utils import decomposite_weight_matric_with_rank, swap_shift_scale_for_linear_weight


def test_swap_shift_scale_for_linear_weight():
    """Test that swaping should work"""
    original = torch.Tensor([1, 2])
    expected = torch.Tensor([2, 1])

    swapped = swap_shift_scale_for_linear_weight(original)
    assert(torch.allclose(expected, swapped))

    size= (3, 4)
    first = torch.randn(size)
    second = torch.randn(size)

    original = torch.concat([first, second])
    expected = torch.concat([second, first])

    swapped = swap_shift_scale_for_linear_weight(original)
    assert(torch.allclose(expected, swapped))

    # call this twice will reconstruct the original
    reconstructed = swap_shift_scale_for_linear_weight(swapped)
    assert(torch.allclose(reconstructed, original))

def test_decomposite_weight_matric_with_rank():
    """Test that decompsition of given matrix into 2 low rank matrices work"""
    input_dim = 1024
    output_dim = 1024
    rank = 8  # Low rank


    A = torch.randn(input_dim, rank).double()
    B = torch.randn(rank, output_dim).double()
    W0 = A @ B

    C, D = decomposite_weight_matric_with_rank(W0, rank)
    R = C @ D

    assert(C.shape == A.shape)
    assert(D.shape == B.shape)

    assert torch.allclose(W0, R)

