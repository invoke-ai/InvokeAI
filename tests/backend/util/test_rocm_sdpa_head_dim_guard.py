"""The ROCm SDPA guard for `F.scaled_dot_product_attention`.

The fused ROCm kernels are wrong above head_dim 256 (see `ROCM_FUSED_SDPA_MAX_HEAD_DIM`). The guard
must send exactly those calls to the math kernel, leave every other call on the original function
with its arguments intact, install once, and make the working-memory estimator budget the score
matrix the math kernel materializes. Math-kernel calls whose score matrix exceeds
`SDPA_MATH_CHUNK_BYTES` must run in chunks that each stay within it and together give the unchunked
result.
"""

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend

import invokeai.backend.util.attention as attention
from invokeai.backend.util.attention import (
    ROCM_FUSED_SDPA_MAX_HEAD_DIM,
    install_rocm_sdpa_guard,
    rocm_sdpa_uses_math_kernel,
    sdpa_score_matrix_bytes,
)

WIDE = ROCM_FUSED_SDPA_MAX_HEAD_DIM + 1
NARROW = ROCM_FUSED_SDPA_MAX_HEAD_DIM
BYTES = attention.SDPA_MATH_BYTES_PER_SCORE_ELEMENT
# torch's own function, captured before any test installs the guard: the unchunked oracle.
TORCH_SDPA = F.scaled_dot_product_attention


def _reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.softmax(q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5, dim=-1) @ v


class TestPredicate:
    """The one rule the guard and the estimators share, tested against real device objects."""

    def test_only_wide_heads_on_a_rocm_cuda_device(self, monkeypatch):
        monkeypatch.setattr(attention, "_IS_ROCM", True)
        assert rocm_sdpa_uses_math_kernel("cuda", WIDE)
        assert not rocm_sdpa_uses_math_kernel("cuda", NARROW), "the threshold itself is trusted"
        assert not rocm_sdpa_uses_math_kernel("cpu", WIDE)
        assert not rocm_sdpa_uses_math_kernel("mps", WIDE)

    def test_never_off_rocm(self, monkeypatch):
        monkeypatch.setattr(attention, "_IS_ROCM", False)
        assert not rocm_sdpa_uses_math_kernel("cuda", WIDE)

    def test_threshold_override_from_the_environment(self, monkeypatch):
        monkeypatch.setenv(attention._ROCM_FUSED_SDPA_MAX_HEAD_DIM_ENV, "100000")
        assert attention._read_rocm_fused_sdpa_max_head_dim() == 100000
        monkeypatch.setenv(attention._ROCM_FUSED_SDPA_MAX_HEAD_DIM_ENV, "wide")
        assert attention._read_rocm_fused_sdpa_max_head_dim() == attention._ROCM_FUSED_SDPA_MAX_HEAD_DIM_DEFAULT


def _install_as_if_rocm(monkeypatch) -> list[tuple]:
    """Install the guard on a CPU build, treating CPU tensors as the ROCm device, and record the
    full argument tuple of every call that reaches the original function."""
    calls: list[tuple] = []
    original = F.scaled_dot_product_attention

    def spy(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, *, scale=None, enable_gqa=False):
        calls.append((query.shape[-1], attn_mask, dropout_p, is_causal, scale, enable_gqa))
        return original(query, key, value, attn_mask, dropout_p, is_causal, scale=scale, enable_gqa=enable_gqa)

    monkeypatch.setattr(F, "scaled_dot_product_attention", spy)
    monkeypatch.setattr(attention, "_IS_ROCM", True)
    # CPU tensors stand in for the ROCm device; the real predicate is covered by TestPredicate.
    monkeypatch.setattr(attention, "_rocm_cuda", lambda device_type: True)
    install_rocm_sdpa_guard()
    return calls


class TestGuard:
    def test_wide_heads_take_the_math_kernel_and_match_the_reference(self, monkeypatch):
        calls = _install_as_if_rocm(monkeypatch)
        torch.manual_seed(0)
        q, k, v = (torch.randn(1, 1, 64, ROCM_FUSED_SDPA_MAX_HEAD_DIM * 2) for _ in range(3))

        out = F.scaled_dot_product_attention(q, k, v)

        assert calls == [], "a wide head must not reach the fused entry point"
        torch.testing.assert_close(out, _reference(q, k, v), atol=1e-5, rtol=1e-4)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"attn_mask": torch.rand(16, 16) > 0.3, "scale": 0.5, "enable_gqa": True},
            {"attn_mask": torch.randn(1, 1, 16, 16), "enable_gqa": True},
            {"attn_mask": torch.randn(16, 16).to(torch.bfloat16), "enable_gqa": True},
            {"is_causal": True, "enable_gqa": True},
        ],
        ids=["bool-mask+scale", "float-broadcast-mask", "bf16-mask", "causal"],
    )
    def test_wide_heads_keep_torchs_semantics_for_masks_scale_causal_and_gqa(self, monkeypatch, kwargs):
        _install_as_if_rocm(monkeypatch)
        original = F.scaled_dot_product_attention.__wrapped__
        torch.manual_seed(1)
        q = torch.randn(1, 4, 16, WIDE)
        k, v = (torch.randn(1, 2, 16, WIDE) for _ in range(2))
        mask = kwargs.get("attn_mask")
        if mask is not None and mask.dtype == torch.bfloat16:
            q, k, v = (t.to(torch.bfloat16) for t in (q, k, v))

        out = F.scaled_dot_product_attention(q, k, v, **kwargs)

        # torch's own (CPU, unguarded) semantics are the oracle.
        expected = original(q, k, v, **kwargs)
        torch.testing.assert_close(out, expected, atol=1e-2 if out.dtype == torch.bfloat16 else 1e-5, rtol=1e-2)

    def test_narrow_heads_pass_through_with_every_argument(self, monkeypatch):
        calls = _install_as_if_rocm(monkeypatch)
        torch.manual_seed(2)
        q, k, v = (torch.randn(1, 2, 8, NARROW) for _ in range(3))
        mask = torch.ones(8, 8, dtype=torch.bool)

        F.scaled_dot_product_attention(q, k, v, mask, 0.0, True, scale=0.25)

        assert calls == [(NARROW, mask, 0.0, True, 0.25, False)]

    def test_wide_heads_refuse_arguments_the_math_call_cannot_forward(self, monkeypatch):
        _install_as_if_rocm(monkeypatch)
        q = torch.randn(1, 1, 8, WIDE)
        with pytest.raises(TypeError, match="does not forward"):
            F.scaled_dot_product_attention(q, q, q, some_future_flag=True)

    def test_install_is_idempotent_and_inert_off_rocm(self, monkeypatch):
        _install_as_if_rocm(monkeypatch)
        guarded = F.scaled_dot_product_attention
        install_rocm_sdpa_guard()
        assert F.scaled_dot_product_attention is guarded, "a second install must not wrap the wrapper"

        monkeypatch.setattr(F, "scaled_dot_product_attention", guarded.__wrapped__)
        monkeypatch.setattr(attention, "_IS_ROCM", False)
        install_rocm_sdpa_guard()
        assert F.scaled_dot_product_attention is guarded.__wrapped__, "no guard on a non-ROCm build"

    def test_install_backs_off_when_the_math_op_rejects_the_probe(self, monkeypatch, caplog):
        """A torch whose aten op changed must fail at startup, visibly, not inside a generation."""
        original = F.scaled_dot_product_attention
        monkeypatch.setattr(attention, "_IS_ROCM", True)
        monkeypatch.setattr(attention, "_math_sdpa", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no such op")))
        caplog.set_level("WARNING", logger=attention.__name__)

        install_rocm_sdpa_guard()

        assert F.scaled_dot_product_attention is original
        assert any("NOT installed" in r.message for r in caplog.records)

    def test_apply_monkeypatches_installs_the_guard(self, monkeypatch):
        from invokeai.app.util.startup_utils import apply_monkeypatches

        original = F.scaled_dot_product_attention
        # Pin both entry points apply_monkeypatches rebinds so monkeypatch restores them afterwards.
        monkeypatch.setattr(F, "scaled_dot_product_attention", original)
        monkeypatch.setattr(torch.cuda, "empty_cache", torch.cuda.empty_cache)
        monkeypatch.setattr(attention, "_IS_ROCM", True)

        apply_monkeypatches()

        assert getattr(F.scaled_dot_product_attention, attention._ROCM_SDPA_GUARD_SENTINEL, False)


def _install_chunking(
    monkeypatch, budget: int, materializes: bool | None = True, torch_error: str | None = None
) -> list[tuple[torch.Size, torch.Size]]:
    """Install the guard as if on ROCm with a small chunk budget; record (query, key) shapes of every call that reaches
    torch's function or the math op. ``materializes`` pins the dispatch probe (None leaves it real); ``torch_error``
    makes torch's function raise it, as a build without an eligible kernel does."""
    shapes: list[tuple[torch.Size, torch.Size]] = []

    def spy(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, *, scale=None, enable_gqa=False):
        shapes.append((query.shape, key.shape))
        if torch_error is not None:
            raise RuntimeError(torch_error)
        return TORCH_SDPA(query, key, value, attn_mask, dropout_p, is_causal, scale=scale, enable_gqa=enable_gqa)

    real_math = attention._math_sdpa

    def math_spy(query, key, value, *args):
        shapes.append((query.shape, key.shape))
        return real_math(query, key, value, *args)

    monkeypatch.setattr(F, "scaled_dot_product_attention", spy)
    monkeypatch.setattr(attention, "_math_sdpa", math_spy)
    monkeypatch.setattr(attention, "_IS_ROCM", True)
    monkeypatch.setattr(attention, "_rocm_cuda", lambda device_type: True)
    monkeypatch.setattr(attention, "SDPA_MATH_CHUNK_BYTES", budget)
    if materializes is not None:
        monkeypatch.setattr(attention, "_materializes", lambda *args: materializes)
    install_rocm_sdpa_guard()
    shapes.clear()  # the install probe's own math call
    return shapes


def _scores(shape_pair: tuple[torch.Size, torch.Size]) -> int:
    query_shape, key_shape = shape_pair
    return query_shape[0] * query_shape[1] * query_shape[2] * key_shape[-2] * BYTES


class TestChunking:
    """Math-kernel attention over the budget runs in chunks; every chunk stays within it, together they are exact."""

    def test_heads_are_split_first_and_the_result_matches_the_unchunked_call(self, monkeypatch):
        """Bitwise here because the CPU kernel's arithmetic does not depend on the batch it runs over; on a ROCm
        card the batched GEMM re-tiles with the group count and lands within a bf16 ulp instead."""
        per_head = 64 * 48 * BYTES
        shapes = _install_chunking(monkeypatch, budget=2 * per_head)
        torch.manual_seed(0)
        q = torch.randn(1, 6, 64, 32)
        k, v = torch.randn(1, 6, 48, 32), torch.randn(1, 6, 48, 32)

        out = F.scaled_dot_product_attention(q, k, v)

        assert [s[0][1] for s in shapes] == [2, 2, 2], "three head groups, each with every query row"
        assert torch.equal(out, TORCH_SDPA(q, k, v))

    def test_a_single_oversized_head_is_split_by_query_rows(self, monkeypatch):
        """The one-head VAE mid-block: only query rows are left to split, on the wide-head math path."""
        per_row = 96 * BYTES
        shapes = _install_chunking(monkeypatch, budget=20 * per_row)
        torch.manual_seed(1)
        q, k, v = (torch.randn(1, 1, 96, WIDE) for _ in range(3))

        out = F.scaled_dot_product_attention(q, k, v)

        assert [s[0][2] for s in shapes] == [20, 20, 20, 20, 16]
        torch.testing.assert_close(out, _reference(q, k, v), atol=1e-5, rtol=1e-4)

    @pytest.mark.parametrize(
        ("heads", "kv_heads", "budget_heads", "kwargs"),
        [
            (4, 4, 2, {"attn_mask": torch.rand(64, 48) > 0.3}),
            (4, 4, 2, {"attn_mask": torch.randn(2, 1, 64, 48)}),
            (4, 4, 2, {"attn_mask": torch.randn(2, 4, 64, 48)}),
            (4, 4, 2, {"attn_mask": torch.rand(1, 1, 1, 48) > 0.2}),
            (4, 4, 2, {"attn_mask": torch.randn(4, 64, 48)}),
            (4, 4, 2, {"scale": 0.3}),
            (6, 2, 3, {"enable_gqa": True}),
            (6, 2, 2, {"enable_gqa": True}),
            (4, 1, 2, {}),
            (4, 4, 0.5, {"attn_mask": torch.rand(64, 48) > 0.3}),
            (4, 4, 0.5, {"attn_mask": torch.randn(2, 1, 64, 48)}),
            (4, 4, 0.5, {"attn_mask": torch.randn(2, 4, 64, 48)}),
            (4, 4, 0.5, {"attn_mask": torch.rand(2, 1, 1, 48) > 0.2}),
        ],
        ids=[
            "bool-2d",
            "additive-broadcast-heads",
            "additive-per-head",
            "bool-broadcast-rows",
            "3d-per-head",
            "scale",
            "gqa-whole-groups",
            "gqa-group-over-budget",
            "kv-broadcast",
            "rows-bool-2d",
            "rows-additive-broadcast-heads",
            "rows-additive-per-head",
            "rows-bool-broadcast-rows",
        ],
    )
    def test_masks_scale_and_grouped_heads_keep_torchs_result(self, monkeypatch, heads, kv_heads, budget_heads, kwargs):
        budget = int(budget_heads * 2 * 64 * 48 * BYTES)  # under one head, the query rows are split
        shapes = _install_chunking(monkeypatch, budget=budget)
        torch.manual_seed(2)
        q = torch.randn(2, heads, 64, 32)
        k, v = torch.randn(2, kv_heads, 48, 32), torch.randn(2, kv_heads, 48, 32)

        out = F.scaled_dot_product_attention(q, k, v, **kwargs)

        assert len(shapes) > 1, "the call must actually have been split"
        assert all(_scores(pair) <= budget for pair in shapes), "a chunk exceeded the budget"
        torch.testing.assert_close(out, TORCH_SDPA(q, k, v, **kwargs), atol=1e-5, rtol=1e-4)

    @pytest.mark.parametrize(
        ("q_shape", "kv_shape"),
        [((2, 4, 64, 32), (4, 48, 32)), ((1, 4, 64, 32), (2, 4, 48, 32))],
        ids=["3d-kv", "kv-wider-batch"],
    )
    def test_kv_broadcasts_the_chunks_cannot_slice_stay_whole(self, monkeypatch, q_shape, kv_shape):
        shapes = _install_chunking(monkeypatch, budget=BYTES)
        torch.manual_seed(3)
        q, k, v = torch.randn(q_shape), torch.randn(kv_shape), torch.randn(kv_shape)

        out = F.scaled_dot_product_attention(q, k, v)

        assert len(shapes) == 1
        torch.testing.assert_close(out, TORCH_SDPA(q, k, v))

    @pytest.mark.parametrize("kwargs", [{"is_causal": True}, {"dropout_p": 0.1}], ids=["causal", "dropout"])
    def test_causal_and_dropout_calls_stay_whole(self, monkeypatch, kwargs):
        shapes = _install_chunking(monkeypatch, budget=BYTES)
        q = torch.randn(1, 2, 16, 8)

        F.scaled_dot_product_attention(q, q, q, **kwargs)

        assert shapes == [(q.shape, q.shape)]

    def test_a_call_torch_serves_with_a_fused_kernel_stays_whole(self, monkeypatch):
        shapes = _install_chunking(monkeypatch, budget=BYTES, materializes=None)
        monkeypatch.setattr(torch.ops.aten, "_fused_sdp_choice", lambda *a, **k: int(SDPBackend.FLASH_ATTENTION))
        q = torch.randn(1, 2, 16, 8)

        F.scaled_dot_product_attention(q, q, q)

        assert shapes == [(q.shape, q.shape)]

    def test_no_eligible_kernel_raises_the_same_error_as_the_whole_call(self, monkeypatch):
        """The probe raising means torch has no backend for the call (an exclusive policy, say): chunking must neither
        hide that nor route around it onto a kernel the policy forbids."""
        message = "No available kernel. Aborting execution."
        shapes = _install_chunking(monkeypatch, budget=16 * 16 * BYTES, materializes=None, torch_error=message)

        def no_backend(*args, **kwargs):
            raise RuntimeError("No viable backend for scaled_dot_product_attention was found.")

        monkeypatch.setattr(torch.ops.aten, "_fused_sdp_choice", no_backend)
        q = torch.randn(1, 2, 16, 8)

        with pytest.raises(RuntimeError, match=message):
            F.scaled_dot_product_attention(q, q, q)
        assert shapes[0][0][1] == 1, "the first chunk, run through torch's own function, raised"

    def test_the_output_dtype_follows_autocast(self, monkeypatch):
        per_head = 32 * 32 * BYTES
        _install_chunking(monkeypatch, budget=per_head)
        q = torch.randn(1, 4, 32, 16)

        with torch.autocast("cpu", dtype=torch.bfloat16):
            chunked = F.scaled_dot_product_attention(q, q, q)
            whole = TORCH_SDPA(q, q, q)

        assert chunked.dtype == whole.dtype
        torch.testing.assert_close(chunked.float(), whole.float(), atol=1e-2, rtol=1e-2)


class TestEstimator:
    def test_wide_heads_on_rocm_reserve_the_score_matrix_without_asking_torch(self, monkeypatch):
        monkeypatch.setattr(attention, "_IS_ROCM", True)
        probed: list[int] = []

        def fused_choice(q, *args, **kwargs):
            probed.append(q.shape[-1])
            return next(iter(attention._FUSED_SDP_CHOICES))

        monkeypatch.setattr(torch.ops.aten, "_fused_sdp_choice", fused_choice)
        # On a CPU-only build the probe cannot even allocate on "cuda" and would report the
        # materializing path for the wrong reason; make it run so the test discriminates everywhere.
        real_empty = torch.empty
        monkeypatch.setattr(torch, "empty", lambda *a, device=None, **kw: real_empty(*a, **kw))
        device = torch.device("cuda", 0)

        wide = sdpa_score_matrix_bytes(device=device, dtype=torch.bfloat16, num_heads=1, head_dim=WIDE, seq_len=64)
        assert wide == 64 * 64 * attention.SDPA_MATH_BYTES_PER_SCORE_ELEMENT
        assert probed == []

        narrow = sdpa_score_matrix_bytes(device=device, dtype=torch.bfloat16, num_heads=1, head_dim=NARROW, seq_len=64)
        assert narrow == 0, "a narrow head still asks torch, and a fused answer means no score matrix"
        assert probed == [NARROW]

    def test_wide_heads_off_rocm_still_ask_torch(self, monkeypatch):
        monkeypatch.setattr(attention, "_IS_ROCM", False)
        monkeypatch.setattr(
            torch.ops.aten, "_fused_sdp_choice", lambda q, *a, **k: next(iter(attention._FUSED_SDP_CHOICES))
        )
        assert (
            sdpa_score_matrix_bytes(
                device=torch.device("cpu"), dtype=torch.float32, num_heads=1, head_dim=WIDE, seq_len=64
            )
            == 0
        )

    def test_the_estimate_is_capped_at_one_chunk_only_while_the_guard_chunks(self, monkeypatch):
        """With the guard installed, one chunk is alive at a time; without it -- install failed, or a script that
        never installs it -- the whole matrix is built and must be budgeted in full."""
        monkeypatch.setattr(attention, "_IS_ROCM", True)
        monkeypatch.setattr(F, "scaled_dot_product_attention", TORCH_SDPA)
        device = torch.device("cuda", 0)
        full = 16384 * 16384 * BYTES

        def estimate() -> int:
            return sdpa_score_matrix_bytes(
                device=device, dtype=torch.bfloat16, num_heads=1, head_dim=WIDE, seq_len=16384
            )

        assert estimate() == full

        install_rocm_sdpa_guard()
        assert estimate() == attention.SDPA_MATH_CHUNK_BYTES
        assert (
            sdpa_score_matrix_bytes(device=device, dtype=torch.bfloat16, num_heads=1, head_dim=WIDE, seq_len=64)
            == 64 * 64 * BYTES
        ), "a matrix within one chunk is priced as it is"


needs_rocm = pytest.mark.skipif(torch.version.hip is None or not torch.cuda.is_available(), reason="needs a ROCm GPU")


@pytest.mark.slow
@needs_rocm
class TestOnRocmHardware:
    """The upstream defect and the guard, on the real device. `slow`: a dev-machine lane, not CI."""

    SHAPE = (1, 1, 16384, 512)  # the FLUX.2 VAE mid-block at a 1024px decode

    def _inputs(self):
        torch.manual_seed(0)
        return (torch.randn(self.SHAPE, device="cuda", dtype=torch.bfloat16) for _ in range(3))

    def test_a_fused_kernel_is_still_wrong_for_the_wide_head(self):
        """Pins the reason the guard exists. When this passes on a new torch, the guard can go."""
        from torch.nn.attention import SDPBackend

        from invokeai.backend.util.sdpa_scope import sdpa_policy

        q, k, v = self._inputs()
        reference = attention._math_sdpa(q, k, v, None, 0.0, False, None, False).float()
        # The unguarded entry point, whether or not the guard is installed in this process.
        unguarded = getattr(F.scaled_dot_product_attention, "__wrapped__", F.scaled_dot_product_attention)
        wrong = ran = 0
        for backend in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION):
            with sdpa_policy([backend]):
                try:
                    out = unguarded(q, k, v).float()
                except RuntimeError:
                    continue  # the backend refused the shape: not wrong, just absent
            ran += 1
            err = (out - reference).abs().max().item() / reference.abs().max().item()
            wrong += int(not torch.isfinite(out).all() or err > 0.05)
        if not ran:
            pytest.skip("no fused kernel runs this shape on this build (e.g. gfx1200)")
        assert wrong > 0, "every fused kernel now agrees with math: the ROCm head-dim guard may be unnecessary"

    def test_the_guard_is_exact_under_allocation_churn(self, monkeypatch):
        """The fused kernels' failure depended on allocator state; the guarded path must not. The shape is over one
        chunk, so the reference is the guarded result itself (query chunks differ from one unchunked call by up to a
        bf16 ulp) -- what must hold is that it never changes."""
        import random

        monkeypatch.setattr(F, "scaled_dot_product_attention", F.scaled_dot_product_attention)
        install_rocm_sdpa_guard()
        q, k, v = self._inputs()
        reference = F.scaled_dot_product_attention(q, k, v)
        torch.testing.assert_close(
            reference.float(), attention._math_sdpa(q, k, v, None, 0.0, False, None, False).float(), atol=2e-3, rtol=0
        )
        rnd = random.Random(0)
        for _ in range(20):
            junk = [
                torch.empty(rnd.randint(1, 400) * 2**20, device="cuda", dtype=torch.uint8)
                for _ in range(rnd.randint(1, 4))
            ]
            out = F.scaled_dot_product_attention(q, k, v)
            del junk
            if rnd.random() < 0.5:
                torch.cuda.empty_cache()
            assert torch.equal(out, reference)

    @pytest.mark.parametrize(
        "shape",
        [(1, 30, 4128, 128), (1, 1, 36864, 512)],
        ids=["z-image-1024", "flux-vae-mid-block-1536"],
    )
    def test_math_attention_peaks_within_one_chunk(self, monkeypatch, shape):
        """Z-Image at 1024px would build 8 GiB of scores, a 1536px VAE decode 21 GiB. Where torch has a fused kernel
        for the narrow head, the call is not chunked and peaks far lower anyway."""
        monkeypatch.setattr(F, "scaled_dot_product_attention", F.scaled_dot_product_attention)
        install_rocm_sdpa_guard()
        torch.manual_seed(0)
        q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3))
        F.scaled_dot_product_attention(q, k, v)  # warm-up: kernel selection and allocator growth
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        out = F.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()

        peak = torch.cuda.max_memory_allocated() - base
        assert peak <= attention.SDPA_MATH_CHUNK_BYTES + out.numel() * out.element_size()


class TestClassicVaeEstimators:
    """The SD/SDXL and FLUX.1 VAE estimators price the mid-block score matrix the guard forces."""

    @pytest.fixture(autouse=True)
    def _rocm_cuda(self, monkeypatch):
        """The guard's ROCm switch and the convolution backend are separate questions, and this class is about the
        first: the assertions below carry the cuDNN linear term, so pin `torch.version.hip` (which selects MIOpen's
        larger constants) away from it while the attention side stays on ROCm."""
        import invokeai.backend.util.vae_working_memory as vwm

        monkeypatch.setattr(attention, "_IS_ROCM", True)
        monkeypatch.setattr(torch.version, "hip", None)
        monkeypatch.setattr(vwm.TorchDevice, "choose_torch_device", classmethod(lambda cls: torch.device("cuda", 0)))

    @staticmethod
    def _vae(spec):
        from unittest.mock import MagicMock

        vae = MagicMock(spec=spec)
        vae.parameters.side_effect = lambda: iter([torch.zeros(1, dtype=torch.bfloat16)])
        return vae

    @staticmethod
    def _score_bytes(px: int) -> int:
        tokens = (px // 8) ** 2
        return tokens * tokens * attention.SDPA_MATH_BYTES_PER_SCORE_ELEMENT

    def test_sdxl_untiled_decode_reserves_the_score_matrix_when_it_dominates(self, monkeypatch):
        from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

        from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_sd15_sdxl

        latents = torch.zeros(1, 4, 2048 // 8, 2048 // 8)
        linear = 2048 * 2048 * 2 * 2200
        with_guard = estimate_vae_working_memory_sd15_sdxl(
            "decode", latents, self._vae(AutoencoderKL), None, fp32=False
        )
        assert with_guard == self._score_bytes(2048) > linear

        monkeypatch.setattr(attention, "_IS_ROCM", False)
        monkeypatch.setattr(
            torch.ops.aten, "_fused_sdp_choice", lambda q, *a, **k: next(iter(attention._FUSED_SDP_CHOICES))
        )
        real_empty = torch.empty
        monkeypatch.setattr(torch, "empty", lambda *a, device=None, **kw: real_empty(*a, **kw))
        assert (
            estimate_vae_working_memory_sd15_sdxl("decode", latents, self._vae(AutoencoderKL), None, fp32=False)
            == linear
        )

    @pytest.mark.parametrize("estimator", ["estimate_vae_working_memory_sd3", "estimate_vae_working_memory_cogview4"])
    def test_the_other_autoencoder_kl_estimators_price_it_too(self, estimator):
        from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

        import invokeai.backend.util.vae_working_memory as vwm

        latents = torch.zeros(1, 16, 2048 // 8, 2048 // 8)
        assert getattr(vwm, estimator)("decode", latents, self._vae(AutoencoderKL)) == self._score_bytes(2048)

    def test_a_tiny_autoencoder_has_no_attention_to_price(self):
        from diffusers.models.autoencoders.autoencoder_tiny import AutoencoderTiny

        from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_sd15_sdxl

        latents = torch.zeros(1, 4, 2048 // 8, 2048 // 8)
        assert estimate_vae_working_memory_sd15_sdxl(
            "decode", latents, self._vae(AutoencoderTiny), None, fp32=False
        ) == (2048 * 2048 * 2 * 2200)

    def test_flux1_decode_prices_the_score_matrix_by_tile_when_tiled(self):
        from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

        from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_flux

        latents = torch.zeros(1, 16, 2048 // 8, 2048 // 8)
        untiled = estimate_vae_working_memory_flux("decode", latents, self._vae(AutoencoderKL))
        assert untiled == self._score_bytes(2048)

        tiled = estimate_vae_working_memory_flux("decode", latents, self._vae(AutoencoderKL), tile_size=1024)
        assert tiled < self._score_bytes(2048)
        assert tiled >= self._score_bytes(1024), "a 1024px tile still materializes its own 16384-token score matrix"
