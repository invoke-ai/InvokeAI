"""Whether the K/V heads are expanded before SDPA, or passed through as grouped-query attention.

Krea-2 has 48 query heads over 12 K/V heads. `repeat_interleave` makes them match, which lets any
fused kernel take the call — at the price of allocating two tensors four times larger, every call.
Only some kernels serve the unexpanded shape, so the choice is made per call by asking the
dispatcher, keeping the heads grouped only where a fused kernel takes them -- and, with another
kernel than flash pinned first, only where that kernel ranks no lower than the expanded call's.
Getting that answer wrong in the optimistic direction is not a slow path but a ~9 GB one -- the math
backend materialises the full [heads, seq, seq] score matrix -- or a kernel the admin pinned that
never gets the call.
"""

import contextlib
from unittest.mock import MagicMock

import pytest
import torch
from diffusers.models.transformers.transformer_krea2 import Krea2Attention
from torch.nn.attention import SDPBackend

from invokeai.backend.krea2 import attention as attention_module
from invokeai.backend.krea2.attention import (
    Krea2MemoryEfficientAttnProcessor,
    Krea2RegionalPromptingState,
    Krea2SdpaBackends,
)
from invokeai.backend.util.attention import sdpa_backend_choice
from invokeai.backend.util.sdpa_scope import demote_cudnn_attention, install_default_sdpa_priority, sdpa_policy

HQ, HKV, HEAD_DIM = 48, 12, 128

FLASH, CUDNN, EFFICIENT, MATH, OVERRIDEABLE = (
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
    SDPBackend.OVERRIDEABLE,
)


def _tensors(device: str, seq: int = 64, kv_heads: int = HKV, masked: bool = False):
    q = torch.randn(1, HQ, seq, HEAD_DIM, device=device, dtype=torch.bfloat16)
    kv = torch.randn(1, kv_heads, seq, HEAD_DIM, device=device, dtype=torch.bfloat16)
    mask = torch.zeros(1, 1, seq, seq, device=device, dtype=torch.bfloat16) if masked else None
    return q, kv, kv.clone(), mask


def _fake_cuda(monkeypatch) -> None:
    """Get past the processor's CUDA gate on a runner without a GPU; the dispatcher then answers for CPU."""
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True), raising=False)


class TestTheDecisionIsConservative:
    def test_a_non_cuda_tensor_never_takes_the_gqa_path(self):
        """The grouped path is measured on CUDA only. CPU's own dispatcher would say yes -- its flash
        kernel serves grouped-query attention -- so this pins the gate itself."""
        proc = Krea2MemoryEfficientAttnProcessor()
        assert proc._serves_grouped_query_attention(*_tensors("cpu")) is False


class TestTheHeadsAreOnlyPassedThroughWhenTheyDiffer:
    def test_a_block_with_equal_head_counts_never_asks_and_never_sets_the_flag(self, monkeypatch):
        """Krea-2's text-fusion blocks have num_heads == num_kv_heads, so there is nothing to expand
        and nothing to ask. Asking anyway would pay the probe per call for a `False` the shapes
        already settle -- and setting `enable_gqa` on equal head counts is a claim about the tensors
        that is not this processor's to make."""
        asked = MagicMock(return_value=SDPBackend.FLASH_ATTENTION)
        monkeypatch.setattr(attention_module, "sdpa_runs_fused", asked)
        monkeypatch.setattr(attention_module, "sdpa_backend_choice", asked)

        seen: dict[str, object] = {}
        real_sdpa = attention_module.F.scaled_dot_product_attention

        def spy(query, key, value, attn_mask=None, enable_gqa=False, **kwargs):
            seen["enable_gqa"] = enable_gqa
            seen["kv_heads"] = key.shape[1]
            return real_sdpa(query, key, value, attn_mask=attn_mask, enable_gqa=enable_gqa, **kwargs)

        monkeypatch.setattr(attention_module.F, "scaled_dot_product_attention", spy)

        attn = Krea2Attention(hidden_size=HQ * 16, num_heads=HQ, num_kv_heads=HQ, eps=1e-5).eval()
        attn.set_processor(Krea2MemoryEfficientAttnProcessor())
        with torch.no_grad():
            attn(torch.randn(1, 24, attn.hidden_size), attention_mask=None, image_rotary_emb=None)

        assert seen["enable_gqa"] is False
        assert seen["kv_heads"] == HQ
        assert asked.call_count == 0


class TestTheAnswerIsNotCached:
    """The answer turns on process-global state a cache key cannot enumerate.

    The dispatcher reads the per-backend enable flags and the priority order, which any `sdpa_policy`
    window in the process can move. A cached "no" costs the expansion this function exists to skip for
    the rest of the generation, with nothing logged -- the same reasoning `backend.util.attention`
    records for its own dispatcher probe.
    """

    def test_the_answer_follows_the_selection_in_force(
        self, monkeypatch, cpu_flash_serves_grouped_query_attention, preserve_sdpa_priority_order
    ):
        """Against the real dispatcher, under real windows. A window here stands in for this
        processor's exclusive override and for another session's alike: the dispatcher cannot tell
        them apart, and neither should the answer. Under the default order, which decides with one
        question; a pinned order asks more (`TestAPinnedKernel`).

        Where this torch's CPU flash kernel does not take grouped-query attention (torch 2.7), the K/V
        heads match the query's instead: mismatched, every answer would be no, and the test would no
        longer see the window at all."""
        install_default_sdpa_priority()
        proc = Krea2MemoryEfficientAttnProcessor()
        q, k, v, mask = _tensors("cpu", kv_heads=HKV if cpu_flash_serves_grouped_query_attention else HQ)
        _fake_cuda(monkeypatch)

        assert proc._serves_grouped_query_attention(q, k, v, mask) is True
        with sdpa_policy([SDPBackend.MATH]):
            # Math's grouped-query path expands K/V itself on every call (and every chunk the ROCm guard cuts),
            # where one explicit expansion copies once and math copies nothing for equal heads.
            assert proc._serves_grouped_query_attention(q, k, v, mask) is False
        with sdpa_policy([SDPBackend.EFFICIENT_ATTENTION]):
            # Nothing this window permits serves the call -- the `efficient` override's case. Answering
            # yes would fail it outright; no sends it to the expansion, which efficient takes.
            assert proc._serves_grouped_query_attention(q, k, v, mask) is False
        assert proc._serves_grouped_query_attention(q, k, v, mask) is True

    @pytest.mark.parametrize(
        ("pinned", "questions"),
        [(None, 1), (SDPBackend.EFFICIENT_ATTENTION, 2)],
        ids=["default-order", "efficient-pinned"],
    )
    def test_the_dispatcher_is_asked_under_the_policy_the_call_will_run_with(
        self, monkeypatch, preserve_sdpa_priority_order, pinned, questions
    ):
        """Asked outside the window, the dispatcher answers for whatever policy is installed at that
        moment -- which on a busy multi-GPU box is another session's, not this call's. Under the default
        order that is one question, as it always was; with another kernel pinned first two: the grouped
        call's kernel, and -- cuDNN not ranking first -- the expanded call's."""
        from torch.nn.attention import _cur_sdpa_kernel_backends

        install_default_sdpa_priority(pinned)
        exclusive = Krea2SdpaBackends(exclusive=SDPBackend.CUDNN_ATTENTION)
        proc = Krea2MemoryEfficientAttnProcessor(sdpa_backends=exclusive)
        enabled_when_asked: list[set[SDPBackend]] = []

        def choice(*args, **kwargs):
            enabled_when_asked.append(set(_cur_sdpa_kernel_backends()))
            return SDPBackend.CUDNN_ATTENTION

        monkeypatch.setattr(attention_module, "sdpa_backend_choice", choice)
        monkeypatch.setattr(attention_module, "sdpa_runs_fused", lambda *args, **kwargs: choice() is not None)
        _fake_cuda(monkeypatch)

        attn = Krea2Attention(hidden_size=HQ * 16, num_heads=HQ, num_kv_heads=HKV, eps=1e-5).eval()
        attn.set_processor(proc)
        # The call itself cannot complete: an exclusive cuDNN policy leaves a CPU tensor with no
        # viable kernel. That is downstream of what is being asserted -- the probe has already run by
        # then, and moving it back outside the window makes the assertion below fail either way.
        with torch.no_grad(), contextlib.suppress(RuntimeError):
            attn(torch.randn(1, 24, attn.hidden_size), attention_mask=None, image_rotary_emb=None)

        # Outside the window every backend is enabled, so this set is only reachable from inside it.
        assert enabled_when_asked == [{SDPBackend.CUDNN_ATTENTION}] * questions


class TestTheExpansionStillHappensWhenItMust:
    @pytest.mark.parametrize("masked", [False, True])
    def test_cpu_output_matches_the_stock_processor(self, masked):
        """On CPU the answer is always no, so this is the expansion path — and it must stay exactly
        what it was before the grouped-query option existed."""
        torch.manual_seed(0)
        attn = Krea2Attention(hidden_size=256, num_heads=8, num_kv_heads=2, eps=1e-5).eval()
        hidden = torch.randn(1, 16, 256)
        state = Krea2RegionalPromptingState(attention_mask=torch.ones(16, 16, dtype=torch.bool)) if masked else None

        with torch.no_grad():
            attn.set_processor(Krea2MemoryEfficientAttnProcessor(regional_prompting_state=state))
            ours = attn(hidden, attention_mask=None, image_rotary_emb=None)
            from diffusers.models.transformers.transformer_krea2 import Krea2AttnProcessor

            attn.set_processor(Krea2AttnProcessor())
            stock = attn(hidden, attention_mask=None, image_rotary_emb=None)

        assert torch.allclose(ours, stock, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required to reach a fused GQA kernel")
class TestTheTwoPathsAgreeOnCuda:
    @pytest.mark.parametrize("masked", [False, True])
    def test_grouped_query_output_matches_the_expanded_one(self, masked, monkeypatch, preserve_sdpa_priority_order):
        """The saving is only worth having if the result is the same. Anything else would be a
        silent change to every Krea-2 image."""
        # The order a generation runs under; torch's own ranks math above cuDNN, so the grouped call
        # would not reach the kernel that serves it in production.
        install_default_sdpa_priority()
        torch.manual_seed(0)
        attn = (
            Krea2Attention(hidden_size=HQ * HEAD_DIM, num_heads=HQ, num_kv_heads=HKV, eps=1e-5)
            .eval()
            .to("cuda", torch.bfloat16)
        )
        hidden = torch.randn(1, 256, HQ * HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        state = None
        if masked:
            m = torch.zeros(256, 256, device="cuda", dtype=torch.bool)
            m[:128, :128] = True
            m[128:, 128:] = True
            state = Krea2RegionalPromptingState(attention_mask=m)

        # Record the K/V head count that actually reaches SDPA. Without this the test would pass
        # just as happily if both runs expanded, which is exactly what it is meant to rule out.
        real_sdpa = torch.nn.functional.scaled_dot_product_attention
        seen: list[int] = []

        def spy(q, k, v, attn_mask=None, enable_gqa=False, **kwargs):
            seen.append(k.shape[1])
            return real_sdpa(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa, **kwargs)

        def run(force_expand: bool) -> torch.Tensor:
            proc = Krea2MemoryEfficientAttnProcessor(regional_prompting_state=state)
            if force_expand:
                proc._serves_grouped_query_attention = lambda *a, **k: False
            attn.set_processor(proc)
            monkeypatch.setattr(attention_module.F, "scaled_dot_product_attention", spy)
            with torch.no_grad():
                out = attn(hidden, attention_mask=None, image_rotary_emb=None).float()
            monkeypatch.undo()
            return out

        expanded = run(True)
        grouped = run(False)

        assert seen[0] == HQ, "the forced run should have expanded the K/V heads"

        # Skip only where the dispatcher genuinely refuses the shape. Skipping on `seen[1] != HKV`
        # alone turns "the change stopped taking effect" into a green run: an inverted condition or
        # a permanently-False `enable_gqa` restores the per-call expansion, removes the entire
        # benefit, and nothing fails anywhere.
        proc = Krea2MemoryEfficientAttnProcessor(regional_prompting_state=state)
        q, k, v, mask = _tensors("cuda", seq=256, masked=masked)
        if not proc._serves_grouped_query_attention(q, k, v, mask):
            pytest.skip("no fused kernel on this device serves grouped-query attention here")
        assert seen[1] == HKV, "the dispatcher serves this shape, so the grouped path must have run"

        # bf16 kernels differ in accumulation order, so the two are close rather than identical. The
        # bound is the one used for the backend comparison elsewhere.
        corr = torch.corrcoef(torch.stack([expanded.flatten(), grouped.flatten()]))[0, 1]
        assert corr > 0.9999, f"correlation {corr}"
        assert (expanded - grouped).abs().max() < 0.05


@pytest.mark.skipif(not torch.cuda.is_available(), reason="only CUDA's chooser honours the priority order")
class TestTheOrderDecidesToo:
    def test_a_capable_kernel_ranked_below_math_does_not_count(self, preserve_sdpa_priority_order):
        """The ~9 GB case. Under torch's own order -- which is also the order after a cuDNN failure
        demoted it -- math ranks above cuDNN, and with a mask flash refuses the call and
        memory-efficient refuses grouped-query attention, so the unexpanded call would run on math even
        where cuDNN could serve it. Asking only whether a kernel *can* serve the call would answer yes."""
        proc = Krea2MemoryEfficientAttnProcessor()
        q, k, v, mask = _tensors("cuda", seq=256, masked=True)

        install_default_sdpa_priority()
        if not proc._serves_grouped_query_attention(q, k, v, mask):
            pytest.skip("no fused kernel on this device serves masked grouped-query attention")
        assert demote_cudnn_attention() is True
        assert proc._serves_grouped_query_attention(q, k, v, mask) is False


class TestTheClaimHoldsWithoutCuda:
    """The saving rests on grouped output equalling expanded output, and that does not need a GPU.

    `_serves_grouped_query_attention` returns False for CPU tensors, so on a runner without one
    every other test in this file exercises the expansion path and the change itself is unobserved.
    `enable_gqa` is served on CPU by torch's own kernels, so the numerical claim can be pinned there —
    which is the half that would silently change every Krea-2 image if it were wrong.
    """

    @pytest.mark.parametrize("masked", [False, True])
    def test_grouped_and_expanded_sdpa_agree(self, masked):
        torch.manual_seed(0)
        seq = 32
        q = torch.randn(1, HQ, seq, HEAD_DIM, dtype=torch.float32)
        k = torch.randn(1, HKV, seq, HEAD_DIM, dtype=torch.float32)
        v = torch.randn(1, HKV, seq, HEAD_DIM, dtype=torch.float32)
        mask = None
        if masked:
            mask = torch.zeros(1, 1, seq, seq, dtype=torch.float32)
            mask[..., seq // 2 :, : seq // 2] = float("-inf")

        grouped = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        repeats = HQ // HKV
        expanded = torch.nn.functional.scaled_dot_product_attention(
            q, k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1), attn_mask=mask
        )

        assert grouped.shape == expanded.shape
        assert torch.allclose(grouped, expanded, atol=1e-5)

    def test_the_processor_passes_the_grouped_heads_through_when_the_answer_is_yes(
        self, monkeypatch, preserve_sdpa_priority_order
    ):
        """That the decision reaches SDPA — the half the CUDA tests cover only where there is a GPU.

        With the device and the dispatcher's answer faked, the answer is yes, so the processor must
        hand SDPA the unexpanded K/V heads and `enable_gqa=True`. Pinning it here is what makes a
        permanently-False `enable_gqa` fail on a runner without CUDA, where every other test in this
        file legitimately exercises the expansion path instead.
        """
        seen: dict[str, object] = {}
        real = attention_module.F.scaled_dot_product_attention

        def spy(query, key, value, attn_mask=None, enable_gqa=False, **kwargs):
            seen["kv_heads"] = key.shape[1]
            seen["enable_gqa"] = enable_gqa
            return real(query, key, value, attn_mask=attn_mask, enable_gqa=enable_gqa, **kwargs)

        install_default_sdpa_priority()
        _fake_cuda(monkeypatch)
        monkeypatch.setattr(attention_module, "sdpa_runs_fused", lambda *args, **kwargs: True)
        monkeypatch.setattr(attention_module.F, "scaled_dot_product_attention", spy)

        torch.manual_seed(0)
        attn = Krea2Attention(hidden_size=HQ * HEAD_DIM, num_heads=HQ, num_kv_heads=HKV, eps=1e-5).eval()
        attn.set_processor(Krea2MemoryEfficientAttnProcessor())
        with torch.no_grad():
            attn(torch.randn(1, 32, HQ * HEAD_DIM), attention_mask=None, image_rotary_emb=None)

        assert seen["enable_gqa"] is True
        assert seen["kv_heads"] == HKV, "the K/V heads were expanded despite the dispatcher serving them"


PINNED_EFFICIENT = [EFFICIENT, FLASH, CUDNN, MATH, OVERRIDEABLE]
PINNED_CUDNN = [CUDNN, FLASH, EFFICIENT, MATH, OVERRIDEABLE]


def _run_processor(monkeypatch, answer) -> list[tuple[int, bool]]:
    """Run a small grouped-query attention through the processor with the device faked and the dispatcher's answers
    scripted by `answer(enable_gqa)`, and record the K/V heads and `enable_gqa` each SDPA call gets."""
    _fake_cuda(monkeypatch)
    monkeypatch.setattr(
        attention_module, "sdpa_backend_choice", lambda *args, enable_gqa=False, **kwargs: answer(enable_gqa)
    )
    seen: list[tuple[int, bool]] = []
    real = attention_module.F.scaled_dot_product_attention

    def spy(query, key, value, attn_mask=None, enable_gqa=False, **kwargs):
        seen.append((key.shape[1], enable_gqa))
        return real(query, key, value, attn_mask=attn_mask, enable_gqa=enable_gqa, **kwargs)

    monkeypatch.setattr(attention_module.F, "scaled_dot_product_attention", spy)
    torch.manual_seed(0)
    attn = Krea2Attention(hidden_size=256, num_heads=8, num_kv_heads=2, eps=1e-5).eval()
    attn.set_processor(Krea2MemoryEfficientAttnProcessor())
    with torch.no_grad():
        attn(torch.randn(1, 16, 256), attention_mask=None, image_rotary_emb=None)
    return seen


class TestAPinnedKernel:
    """With another kernel than flash first, a fused grouped call keeps the heads grouped only where its kernel ranks no
    lower than the expanded call's.

    The answers are the dispatcher's, per build and GPU; CPU's dispatcher ignores the order, so they are scripted here
    with what each platform gives (measured or documented in `backend/krea2/attention.py`). The order is real.
    """

    @pytest.mark.parametrize(
        ("grouped", "expanded", "order", "keeps_grouped"),
        [
            # Windows: the kernel pinned away from would take the grouped call; the pinned one takes the expanded call.
            pytest.param(CUDNN, EFFICIENT, PINNED_EFFICIENT, False, id="pinned-efficient"),
            # Linux: flash takes the grouped call, still below the pinned kernel.
            pytest.param(FLASH, EFFICIENT, PINNED_EFFICIENT, False, id="pinned-efficient-flash-takes-it-grouped"),
            # A call memory-efficient attention cannot take expanded either: cuDNN either way.
            pytest.param(CUDNN, CUDNN, PINNED_EFFICIENT, True, id="pinned-efficient-cudnn-either-way"),
            # Expanded, the call would build the full score matrix; grouped, a fused kernel takes it.
            pytest.param(CUDNN, MATH, PINNED_EFFICIENT, True, id="pinned-efficient-expanded-on-math"),
            pytest.param(CUDNN, None, PINNED_EFFICIENT, True, id="expanded-call-has-no-kernel"),
            pytest.param(CUDNN, FLASH, PINNED_CUDNN, True, id="pinned-cudnn"),
        ],
    )
    def test_the_grouped_heads_are_kept_only_where_they_lose_no_rank(self, grouped, expanded, order, keeps_grouped):
        assert attention_module._grouped_ranks_first(grouped, expanded, order) is keeps_grouped

    def test_the_processor_expands_the_heads_for_a_pinned_kernel_that_needs_it(
        self, monkeypatch, preserve_sdpa_priority_order
    ):
        """The rule reaching SDPA: as on an RTX 4090 under Windows with `efficient` pinned -- cuDNN for the grouped
        call, memory-efficient attention for the expanded one."""
        install_default_sdpa_priority(SDPBackend.EFFICIENT_ATTENTION)

        seen = _run_processor(monkeypatch, lambda enable_gqa: CUDNN if enable_gqa else EFFICIENT)

        assert seen == [(8, False)], "expanded to the query heads, without enable_gqa"

    def test_a_grouped_call_math_would_serve_is_expanded(self, monkeypatch, preserve_sdpa_priority_order):
        """`attention_backend: math`: math is first and takes every call. Its grouped-query path would copy the
        expanded K/V on every call, and on every chunk where the ROCm guard splits a long one."""
        install_default_sdpa_priority(SDPBackend.MATH)

        seen = _run_processor(monkeypatch, lambda enable_gqa: MATH)

        assert seen == [(8, False)], "expanded to the query heads, without enable_gqa"


class TestTheExpandedCallIsAskedOnAProbe:
    @pytest.mark.parametrize(
        "device",
        ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"))],
    )
    @pytest.mark.parametrize("mask_kind", ["none", "bool", "float", "padding"])
    def test_the_probe_gets_the_answer_the_expanded_call_gets(self, device, mask_kind):
        """Instead of K/V four times larger, a probe of the call's dtype, head_dim, device and mask kind. That is
        only sound where it answers as the real expanded call does, under any selection in force."""
        seq = 256
        query = torch.empty(1, HQ, seq, HEAD_DIM, device=device, dtype=torch.bfloat16)
        expanded = torch.empty(1, HQ, seq, HEAD_DIM, device=device, dtype=torch.bfloat16)
        mask = {
            "none": None,
            "bool": torch.ones(seq, seq, device=device, dtype=torch.bool),
            "float": torch.zeros(1, 1, seq, seq, device=device, dtype=torch.bfloat16),
            "padding": torch.ones(1, 1, 1, seq, device=device, dtype=torch.bool),
        }[mask_kind]

        for selection in (contextlib.nullcontext(), sdpa_policy([SDPBackend.MATH])):
            with selection:
                answer = sdpa_backend_choice(query, expanded, expanded, mask)
                assert answer is not None
                assert attention_module._expanded_call_choice(query, mask) == answer


def _cuda_attention() -> tuple[Krea2Attention, torch.Tensor]:
    torch.manual_seed(0)
    attn = (
        Krea2Attention(hidden_size=HQ * HEAD_DIM, num_heads=HQ, num_kv_heads=HKV, eps=1e-5)
        .eval()
        .to("cuda", torch.bfloat16)
    )
    return attn, torch.randn(1, 256, HQ * HEAD_DIM, device="cuda", dtype=torch.bfloat16)


def _run_on_cuda(attn: Krea2Attention, hidden: torch.Tensor, monkeypatch) -> tuple[torch.Tensor, list[tuple]]:
    """Run the processor; for each SDPA call record the K/V heads, `enable_gqa` and the kernel that serves it."""
    real = attention_module.F.scaled_dot_product_attention
    calls: list[tuple] = []

    def spy(q, k, v, attn_mask=None, enable_gqa=False, **kwargs):
        calls.append((k.shape[1], enable_gqa, sdpa_backend_choice(q, k, v, attn_mask, enable_gqa=enable_gqa)))
        return real(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa, **kwargs)

    attn.set_processor(Krea2MemoryEfficientAttnProcessor())
    with monkeypatch.context() as patch, torch.no_grad():
        patch.setattr(attention_module.F, "scaled_dot_product_attention", spy)
        out = attn(hidden, attention_mask=None, image_rotary_emb=None).float()
    return out, calls


@pytest.mark.skipif(not torch.cuda.is_available(), reason="only CUDA's chooser honours the priority order")
class TestAPinnedKernelGetsTheCall:
    """Measured on an RTX 4090 under Windows; any CUDA GPU with the kernels answers them."""

    def test_pinned_efficient_expands_the_heads_and_runs_on_it(self, monkeypatch, preserve_sdpa_priority_order):
        """Pinning memory-efficient attention is how an admin keeps attention off cuDNN. It refuses grouped-query
        calls, so the heads are expanded for it rather than left to cuDNN -- and the result is the same image."""
        equal = torch.empty(1, HQ, 256, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        params = torch.backends.cuda.SDPAParams(equal, equal, equal, None, 0.0, False, False)
        if not torch.backends.cuda.can_use_efficient_attention(params):
            pytest.skip("this GPU has no memory-efficient kernel for the Krea-2 shape")
        attn, hidden = _cuda_attention()

        install_default_sdpa_priority(SDPBackend.EFFICIENT_ATTENTION)
        pinned, calls = _run_on_cuda(attn, hidden, monkeypatch)
        install_default_sdpa_priority()
        default, _ = _run_on_cuda(attn, hidden, monkeypatch)

        assert calls == [(HQ, False, SDPBackend.EFFICIENT_ATTENTION)]
        # The bound the two-paths comparison above uses: bf16 kernels differ in accumulation order.
        corr = torch.corrcoef(torch.stack([pinned.flatten(), default.flatten()]))[0, 1]
        assert corr > 0.9999, f"correlation {corr}"
        assert (pinned - default).abs().max() < 0.05

    @pytest.mark.skipif(
        torch.backends.cuda.is_flash_attention_available(), reason="with flash in the build, flash leads the order"
    )
    def test_the_default_order_keeps_the_heads_grouped_on_cudnn(self, monkeypatch, preserve_sdpa_priority_order):
        """Windows CUDA builds have no flash, so cuDNN serves the grouped call and the expanded one alike; the
        expansion would only cost memory."""
        q = torch.empty(1, HQ, 256, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        kv = torch.empty(1, HKV, 256, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        if not torch.backends.cuda.can_use_cudnn_attention(
            torch.backends.cuda.SDPAParams(q, kv, kv, None, 0.0, False, True)
        ):
            pytest.skip("this GPU's cuDNN does not serve grouped-query attention")
        attn, hidden = _cuda_attention()

        install_default_sdpa_priority()
        _, calls = _run_on_cuda(attn, hidden, monkeypatch)

        assert calls == [(HKV, True, SDPBackend.CUDNN_ATTENTION)]
