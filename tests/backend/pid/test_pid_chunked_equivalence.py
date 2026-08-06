"""What `pid_memory_optimization` guarantees about the decoded image, at production dimensions.

`test_pixeldit_official.py` pins the chunked `PiTBlock` against the unchunked one at toy size on the
CPU. That is a real assertion about the *math* - and it holds exactly - but it cannot see the shape
of the problem the setting actually has, because the shipped path runs on CUDA under bf16 autocast
(`PiDDecoder.decode` sets `autocast_dtype = torch.bfloat16` for every CUDA decode) with `BL` in the
thousands rather than 8. There, splitting a GEMM into 1024-row slices makes cuBLAS pick different
kernels and reduction orders, and the results differ by bf16 ULPs.

Measured on an RTX 4090 (torch 2.7.1+cu128), production `PiTBlock` dimensions:

    CPU  fp32 (x86-64, MKL),  BL=2048         ->  max|diff| = 0
    CUDA fp32,                BL=4096         ->  max|diff| = 9.5e-07
    CUDA bf16 autocast,       BL=4096..32768  ->  max|diff| = 1.57e-02, mean|diff| = 2.5e-05

Both paths are internally deterministic, so those numbers are systematic, not run noise. The
consequence downstream is an image that differs slightly: ~43 dB PSNR end-to-end, visually
indistinguishable but not reproducible against an unoptimized decode.

The CPU column is *not* a portable guarantee, and this module originally claimed it was. Asserting
`torch.equal` there passed on x86-64/MKL and failed on macOS/Accelerate in CI: splitting a GEMM along
its row dimension can select a micro-kernel with different K-blocking, so bit-exactness is a property
of the BLAS, not of the chunking. What is portable is that chunking only *reassociates* work, so the
contract here is a distance bound on both paths - tight and scaled to the signal on fp32, wider and
absolute under bf16 autocast (relative tolerances are useless there: activations pass through zero,
so `max|rel|` reaches 1e3 on elements whose absolute error is a single bf16 ULP).
"""

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import torch

from invokeai.backend.pid._src.networks.pixeldit_official import PiTBlock
from invokeai.backend.pid.decode import _PID_ACTIVATION_CHUNK_SIZE

# Production PiTBlock geometry, from `_PID_SR4X_BASE` in invokeai/backend/pid/decode.py.
_PIXEL_HIDDEN_SIZE = 16
_PATCH_HIDDEN_SIZE = 1536
_PATCH_SIZE = 16
_NUM_GROUPS = 24
_ATTN_HIDDEN_SIZE = 1152
_ATTN_NUM_GROUPS = 16

# The contract. Measured worst case under bf16 autocast is 1.57e-02 and is stable across image sizes
# and batch sizes; this leaves ~3x headroom so kernel-selection differences on other GPUs stay inside
# it, while a genuine breakage of the chunked path (wrong slice, misassembled output) blows past it
# by orders of magnitude.
_BF16_ABSOLUTE_TOLERANCE = 5e-2
_BF16_MEAN_ABSOLUTE_TOLERANCE = 1e-3

# The CPU bound, relative to the signal. This started out as `torch.equal`, which held on x86-64 with
# MKL and failed on macOS/Accelerate in CI: splitting a GEMM along M can select a different
# micro-kernel whose K-blocking differs, so even fp32 is only reassociation-close, not bit-equal.
#
# Calibration, measured at these dimensions (signal ~5.7, so one fp32 ULP is ~6.8e-07 absolute):
#   correct code, x86-64/MKL          max|diff| = 0
#   attention contribution off by 1e-6  max|diff| = 7.2e-07  (1.3e-07 relative) - one ULP, invisible
#   attention contribution off by 1e-4  max|diff| = 1.0e-05  (1.9e-06 relative) - ~15 ULP
#
# 1e-5 relative is ~84 ULP: comfortably above whatever reassociation any BLAS produces, and four
# orders of magnitude below a structurally broken chunk, which lands at O(signal). What it does *not*
# do is catch a uniform scaling error smaller than ~2e-06 relative - that band is indistinguishable
# from legitimate backend reassociation, which is the lesson macOS taught here, so no portable test
# can claim it. Structural mistakes (wrong slice, dropped row, unassembled output) are the bug class
# this guards, and they are nowhere near that band.
_CPU_RELATIVE_TOLERANCE = 1e-5


@contextmanager
def _assert_chunked_path_is_taken() -> Iterator[None]:
    """Fail if `_forward_chunked` is not reached.

    Not paranoia: the dispatch guard is `BL > chunk_size`, so the original `batch_size=1` case here
    sat exactly *at* 1024 and silently ran unchunked - the test compared the unchunked path against
    itself and passed for the wrong reason. Asserting the path is taken keeps a future change to the
    guard, the chunk size or the test dimensions from quietly emptying this file out again.
    """
    calls: list[int] = []
    original = PiTBlock._forward_chunked

    def _spy(self: PiTBlock, *args: object, **kwargs: object) -> torch.Tensor:
        calls.append(1)
        return original(self, *args, **kwargs)

    PiTBlock._forward_chunked = _spy  # type: ignore[method-assign]
    try:
        yield
    finally:
        PiTBlock._forward_chunked = original  # type: ignore[method-assign]
    assert calls, "the chunked path was never entered - this comparison proves nothing"


def _build_block(device: str) -> PiTBlock:
    torch.manual_seed(0)
    block = PiTBlock(
        pixel_hidden_size=_PIXEL_HIDDEN_SIZE,
        patch_hidden_size=_PATCH_HIDDEN_SIZE,
        patch_size=_PATCH_SIZE,
        num_heads=_NUM_GROUPS,
        mlp_ratio=4.0,
        attn_hidden_size=_ATTN_HIDDEN_SIZE,
        attn_num_heads=_ATTN_NUM_GROUPS,
        rope_mode="ntk_aware",
        rope_ref_grid_h=64,
        rope_ref_grid_w=64,
    ).eval()
    return block.to(device)


def _inputs(image_px: int, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    patch_grid = image_px // _PATCH_SIZE
    patch_batch = batch_size * patch_grid * patch_grid
    torch.manual_seed(1)
    x = torch.randn(patch_batch, _PATCH_SIZE**2, _PIXEL_HIDDEN_SIZE, device=device)
    s_cond = torch.randn(patch_batch, _PATCH_HIDDEN_SIZE, device=device)
    return x, s_cond


@pytest.mark.parametrize(
    ("image_px", "batch_size"),
    [
        (768, 1),  # BL = 2304: chunk boundaries fall inside a single image
        (512, 2),  # BL = 2048: chunk boundaries straddle two images
    ],
)
def test_chunking_matches_the_unchunked_path_on_cpu(image_px: int, batch_size: int) -> None:
    """Chunking splits the adaLN/MLP GEMMs along their row dimension only, so the result must stay
    at floating-point-reassociation distance from the unchunked path - orders of magnitude below the
    signal, where a real breakage (wrong slice, misassembled output) lands.

    Batch > 1 matters because that is where chunk boundaries stop lining up with image boundaries:
    with `BL = B * Hs * Ws` a chunk can straddle two images, and nothing in `_forward_chunked` may
    depend on an image staying inside one chunk.
    """
    block = _build_block("cpu")
    x, s_cond = _inputs(image_px, batch_size, "cpu")

    with torch.no_grad():
        unchunked = block(x, s_cond, image_px, image_px, _PATCH_SIZE)
        with _assert_chunked_path_is_taken():
            chunked = block(
                x, s_cond, image_px, image_px, _PATCH_SIZE, activation_chunk_size=_PID_ACTIVATION_CHUNK_SIZE
            )

    signal = unchunked.abs().max().item()
    difference = (chunked - unchunked).abs().max().item()
    assert difference <= _CPU_RELATIVE_TOLERANCE * signal, (
        f"chunked output drifted by {difference:.3e} against a signal of {signal:.3e}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the divergence only exists on the accelerated path")
@pytest.mark.parametrize(("image_px", "batch_size"), [(1024, 1), (2048, 1), (2048, 2)])
def test_chunking_stays_within_the_documented_tolerance_on_cuda_bf16(image_px: int, batch_size: int) -> None:
    """The shipped path: CUDA + bf16 autocast, BL well above the chunk size.

    Asserting equality here would be asserting something false. Asserting a bound is the honest
    contract, and it is what the docs promise users who enable the setting.
    """
    block = _build_block("cuda")
    x, s_cond = _inputs(image_px, batch_size, "cuda")
    patch_batch = x.shape[0]
    assert patch_batch >= 2 * _PID_ACTIVATION_CHUNK_SIZE, "the chunked path must actually be exercised"

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        unchunked = block(x, s_cond, image_px, image_px, _PATCH_SIZE)
        with _assert_chunked_path_is_taken():
            chunked = block(
                x, s_cond, image_px, image_px, _PATCH_SIZE, activation_chunk_size=_PID_ACTIVATION_CHUNK_SIZE
            )
            chunked_again = block(
                x, s_cond, image_px, image_px, _PATCH_SIZE, activation_chunk_size=_PID_ACTIVATION_CHUNK_SIZE
            )

    # Determinism first: without it, a passing tolerance would prove nothing about the chunking.
    assert torch.equal(chunked, chunked_again)

    difference = (chunked.float() - unchunked.float()).abs()
    assert difference.max().item() <= _BF16_ABSOLUTE_TOLERANCE
    assert difference.mean().item() <= _BF16_MEAN_ABSOLUTE_TOLERANCE
    # And it is genuinely not exact - if this ever starts failing, the divergence is gone and the
    # tolerance contract (plus the "output changes" wording in the docs) should be revisited.
    assert difference.max().item() > 0.0


def test_discriminator_feature_extraction_never_reaches_the_chunked_pixel_blocks() -> None:
    """`PidNet.forward` returns from the `return_features_early` branch before the pixel pathway.

    The chunk size is only handed to `self.pixel_blocks`, so the discriminator/feature path is
    structurally unaffected by the setting. Pinning it here keeps a future refactor from moving the
    early exit below the pixel loop and quietly putting an untested path under the flag.
    """
    import inspect

    from invokeai.backend.pid._src.networks.pid_net import PidNet

    source = inspect.getsource(PidNet.forward)
    early_exit = source.index("return self._unpatchify_features(")
    pixel_loop = source.index("for blk in self.pixel_blocks:")
    chunk_handoff = source.index("activation_chunk_size=activation_chunk_size")

    assert early_exit < pixel_loop < chunk_handoff


def test_context_parallelism_is_unreachable_so_chunking_cannot_interact_with_it() -> None:
    """Chunking assembles compressed tokens in original order before the *unchanged* attention call,
    but under context parallelism attention gathers k/v across ranks — an interaction nothing here
    tests, and which cannot be tested single-GPU.

    It is also unreachable: `enable_context_parallel` is only ever called from
    `pid/_src/models/pixeldit_model.py`, a vendored upstream class InvokeAI never instantiates (the
    loader goes `load_pid_decoder` -> `build_pid_net` -> `PidNet` directly). So `_cp_group` stays
    None on every decode this application performs. Pinned here so that wiring CP up later cannot
    quietly put chunking into an untested regime.
    """
    from pathlib import Path

    import invokeai

    invokeai_root = Path(invokeai.__file__).parent
    vendored = invokeai_root / "backend" / "pid" / "_src"
    # Scan the Python trees only: `invokeai/frontend/web` carries node_modules, which is huge and
    # contains symlinks that break a naive walk.
    callers = [
        path
        for tree in (invokeai_root / "app", invokeai_root / "backend")
        for path in tree.rglob("*.py")
        if vendored not in path.parents and "enable_context_parallel(" in path.read_text(encoding="utf-8")
    ]
    assert callers == [], f"context parallelism is now reachable from {callers}; revisit the chunked attention path"

    assert _build_block("cpu")._cp_group is None
