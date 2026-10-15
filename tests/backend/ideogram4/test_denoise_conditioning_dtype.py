"""The conditioning buffers the Ideogram 4 loop allocates.

They are the largest tensors in the run: `llm_features` is 53248 wide, and the loop holds one buffer
over the whole packed sequence plus a second over the image tokens. The encoder stores them in
float32, so building them in that dtype would cost ~1.8 GB at 1024px for values `forward` casts to
its own dtype on the next line anyway.
"""

import torch

from invokeai.backend.ideogram4.denoise import _compute_dtype, run_ideogram4_denoise
from invokeai.backend.ideogram4.modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer
from invokeai.backend.ideogram4.sampling_utils import LATENT_DIM

# Narrow everywhere the geometry allows: `in_channels` is the packed latent width the loop builds
# its samples with, so it is the released one.
TINY = Ideogram4Config(
    emb_dim=64,
    num_layers=1,
    num_heads=2,
    intermediate_size=128,
    adanln_dim=16,
    in_channels=LATENT_DIM,
    llm_features_dim=32,
    mrope_section=(4, 2, 2),
)


def _recording_transformer(seen: list[torch.dtype], dtype: torch.dtype) -> Ideogram4Transformer:
    model = Ideogram4Transformer(TINY).to(dtype).eval()
    forward = model.forward

    def recording_forward(*, llm_features: torch.Tensor, **kwargs):
        seen.append(llm_features.dtype)
        return forward(llm_features=llm_features, **kwargs)

    model.forward = recording_forward  # type: ignore[method-assign]
    return model


def test_the_loop_hands_the_model_conditioning_in_its_own_dtype() -> None:
    seen: list[torch.dtype] = []
    # Distinguish the model from the float32 encoder without CPU bf16 matmul, which can
    # raise an illegal instruction on Windows runners. This tests allocation, not low-precision arithmetic.
    conditional = _recording_transformer(seen, torch.float64)
    unconditional = _recording_transformer(seen, torch.float64)

    run_ideogram4_denoise(
        conditional_transformer=conditional,
        unconditional_transformer=unconditional,
        llm_features=torch.zeros(4, TINY.llm_features_dim, dtype=torch.float32),
        height=32,
        width=32,
        num_steps=1,
        mu=0.0,
        std=1.0,
        device=torch.device("cpu"),
    )

    # Both branches, and float32 nowhere: the encoder's dtype must not reach the buffers.
    assert seen == [torch.float64, torch.float64]


def test_a_quantized_build_is_read_from_its_compute_dtype_not_its_weights() -> None:
    """nf4 stores uint8 and fp8 storage stores float8; neither is a dtype torch can compute in.

    Both carry the real one on the module, which is what `Ideogram4Transformer.forward` consults
    first -- deriving it from `weight.dtype` instead would build the conditioning in a storage dtype
    and fail in the first matmul.
    """
    model = Ideogram4Transformer(TINY)
    model.input_proj.weight.data = model.input_proj.weight.data.to(torch.float8_e4m3fn)
    model.input_proj.compute_dtype = torch.bfloat16

    assert _compute_dtype(model) is torch.bfloat16

    del model.input_proj.compute_dtype
    assert _compute_dtype(model) is torch.float8_e4m3fn
