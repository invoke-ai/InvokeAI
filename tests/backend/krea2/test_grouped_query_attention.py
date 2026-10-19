"""Whether the K/V heads are expanded before SDPA, or passed through as grouped-query attention.

Krea-2 has 48 query heads over 12 K/V heads. `repeat_interleave` makes them match, which lets any
fused kernel take the call — at the price of allocating two tensors four times larger, every call.
Only some kernels serve the unexpanded shape, so the choice is made per call by asking the
dispatcher. Getting that answer wrong in the optimistic direction is not a slow path but a ~9 GB one:
the math backend materialises the full [heads, seq, seq] score matrix.
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

HQ, HKV, HEAD_DIM = 48, 12, 128


def _tensors(device: str, seq: int = 64, kv_heads: int = HKV, masked: bool = False):
    q = torch.randn(1, HQ, seq, HEAD_DIM, device=device, dtype=torch.bfloat16)
    kv = torch.randn(1, kv_heads, seq, HEAD_DIM, device=device, dtype=torch.bfloat16)
    mask = torch.zeros(1, 1, seq, seq, device=device, dtype=torch.bfloat16) if masked else None
    return q, kv, kv.clone(), mask


class TestTheDecisionIsConservative:
    def test_a_non_cuda_tensor_never_takes_the_gqa_path(self):
        """`SDPAParams` is CUDA-only, and no other backend offers a fused grouped-query path.
        Answering yes here would drop the call onto math."""
        proc = Krea2MemoryEfficientAttnProcessor()
        assert proc._serves_grouped_query_attention(*_tensors("cpu")) is False

    def test_it_answers_no_when_the_permitted_list_excludes_the_capable_kernels(self, monkeypatch):
        """The trap: cuDNN may well serve the shape, but if this call is not permitted to reach it —
        which `INVOKE_KREA2_SDPA_BACKEND=efficient` does — then answering yes makes the call fail
        outright with `No available kernel`, because the memory-efficient kernel refuses GQA.

        Both capability probes are forced to yes, so membership in the permitted list is the only
        thing left that can produce a no. CPU tensors alone would not reach that far: the function
        returns at the `is_cuda` guard, and this assertion would hold with the whole membership test
        deleted. The device is faked the same way `test_a_torch_build_with_a_different_sdpaparams`
        does it, so the branch is genuinely exercised on a runner without a GPU.
        """
        q, k, v, mask = _tensors("cpu")
        monkeypatch.setattr(type(q), "is_cuda", property(lambda self: True), raising=False)
        monkeypatch.setattr(torch.backends.cuda, "SDPAParams", MagicMock(return_value=MagicMock()))
        monkeypatch.setattr(torch.backends.cuda, "can_use_cudnn_attention", lambda params: True)
        monkeypatch.setattr(torch.backends.cuda, "can_use_flash_attention", lambda params: True)

        permitted = Krea2MemoryEfficientAttnProcessor()
        assert permitted._serves_grouped_query_attention(q, k, v, mask) is True

        excluded = Krea2MemoryEfficientAttnProcessor(
            sdpa_backends=Krea2SdpaBackends(backends=(SDPBackend.EFFICIENT_ATTENTION,), set_priority=False)
        )
        assert excluded._serves_grouped_query_attention(q, k, v, mask) is False

    def test_a_torch_build_with_a_different_sdpaparams_answers_no(self, monkeypatch):
        # Falling back to the expansion is correct everywhere; it merely costs memory.
        proc = Krea2MemoryEfficientAttnProcessor()
        q, k, v, mask = _tensors("cpu")
        monkeypatch.setattr(type(q), "is_cuda", property(lambda self: True), raising=False)
        monkeypatch.setattr(torch.backends.cuda, "SDPAParams", MagicMock(side_effect=TypeError("signature changed")))
        assert proc._serves_grouped_query_attention(q, k, v, mask) is False


class TestTheHeadsAreOnlyPassedThroughWhenTheyDiffer:
    def test_a_block_with_equal_head_counts_never_asks_and_never_sets_the_flag(self, monkeypatch):
        """Krea-2's text-fusion blocks have num_heads == num_kv_heads, so there is nothing to expand
        and nothing to ask. Asking anyway would pay the probe per call for a `False` the shapes
        already settle -- and setting `enable_gqa` on equal head counts is a claim about the tensors
        that is not this processor's to make."""
        asked = MagicMock(return_value=True)
        monkeypatch.setattr(torch.backends.cuda, "SDPAParams", MagicMock(return_value=MagicMock()))
        monkeypatch.setattr(torch.backends.cuda, "can_use_cudnn_attention", asked)
        monkeypatch.setattr(torch.backends.cuda, "can_use_flash_attention", asked)

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

    `can_use_*` gates on the per-backend enable flags, which any `sdpa_policy` window in the process
    can move. A cached "no" costs the expansion this function exists to skip for the rest of the
    generation, with nothing logged -- the same reasoning `backend.util.attention` records for its
    own dispatcher probe.
    """

    def test_a_changed_answer_is_seen_rather_than_remembered(self, monkeypatch):
        proc = Krea2MemoryEfficientAttnProcessor()
        q, k, v, mask = _tensors("cpu")
        monkeypatch.setattr(type(q), "is_cuda", property(lambda self: True), raising=False)
        monkeypatch.setattr(torch.backends.cuda, "SDPAParams", MagicMock(return_value=object()))
        monkeypatch.setattr(torch.backends.cuda, "can_use_flash_attention", MagicMock(return_value=False))

        answers = iter([False, True])
        monkeypatch.setattr(torch.backends.cuda, "can_use_cudnn_attention", lambda params: next(answers))

        assert proc._serves_grouped_query_attention(q, k, v, mask) is False
        assert proc._serves_grouped_query_attention(q, k, v, mask) is True

    def test_the_probe_is_asked_under_the_policy_the_call_will_run_with(self, monkeypatch):
        """Asked outside the window, `can_use_*` answers for whatever policy is installed at that
        moment -- which on a busy multi-GPU box is another session's, not this call's."""
        from torch.nn.attention import _cur_sdpa_kernel_backends

        exclusive = Krea2SdpaBackends(backends=(SDPBackend.CUDNN_ATTENTION,), set_priority=False)
        proc = Krea2MemoryEfficientAttnProcessor(sdpa_backends=exclusive)
        enabled_when_asked: list[set[SDPBackend]] = []

        monkeypatch.setattr(torch.backends.cuda, "SDPAParams", MagicMock(return_value=MagicMock()))
        monkeypatch.setattr(
            torch.backends.cuda,
            "can_use_cudnn_attention",
            lambda params: enabled_when_asked.append(set(_cur_sdpa_kernel_backends())) or False,
        )
        monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True), raising=False)

        attn = Krea2Attention(hidden_size=HQ * 16, num_heads=HQ, num_kv_heads=HKV, eps=1e-5).eval()
        attn.set_processor(proc)
        # The call itself cannot complete: an exclusive cuDNN policy leaves a CPU tensor with no
        # viable kernel. That is downstream of what is being asserted -- the probe has already run by
        # then, and moving it back outside the window makes the assertion below fail either way.
        with torch.no_grad(), contextlib.suppress(RuntimeError):
            attn(torch.randn(1, 24, attn.hidden_size), attention_mask=None, image_rotary_emb=None)

        # Outside the window every backend is enabled, so this set is only reachable from inside it.
        assert enabled_when_asked == [{SDPBackend.CUDNN_ATTENTION}]


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
    def test_grouped_query_output_matches_the_expanded_one(self, masked, monkeypatch):
        """The saving is only worth having if the result is the same. Anything else would be a
        silent change to every Krea-2 image."""
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
            pytest.skip("no permitted fused kernel on this device serves grouped-query attention here")
        assert seen[1] == HKV, "the dispatcher serves this shape, so the grouped path must have run"

        # bf16 kernels differ in accumulation order, so the two are close rather than identical. The
        # bound is the one used for the backend comparison elsewhere.
        corr = torch.corrcoef(torch.stack([expanded.flatten(), grouped.flatten()]))[0, 1]
        assert corr > 0.9999, f"correlation {corr}"
        assert (expanded - grouped).abs().max() < 0.05


class TestTheClaimHoldsWithoutCuda:
    """The saving rests on grouped output equalling expanded output, and that does not need a GPU.

    `_serves_grouped_query_attention` returns False for CPU tensors, so on a runner without one
    every other test in this file exercises the expansion path and the change itself is unobserved.
    `enable_gqa` is served on CPU by the math backend, so the numerical claim can be pinned there —
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

    def test_the_processor_passes_the_grouped_heads_through_when_the_answer_is_yes(self, monkeypatch):
        """That the decision reaches SDPA — the half the CUDA tests cover only where there is a GPU.

        With the device and both capability probes faked, the answer is yes, so the processor must
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

        monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True), raising=False)
        monkeypatch.setattr(torch.backends.cuda, "SDPAParams", MagicMock(return_value=MagicMock()))
        monkeypatch.setattr(torch.backends.cuda, "can_use_cudnn_attention", lambda params: True)
        monkeypatch.setattr(torch.backends.cuda, "can_use_flash_attention", lambda params: True)
        monkeypatch.setattr(attention_module.F, "scaled_dot_product_attention", spy)

        torch.manual_seed(0)
        attn = Krea2Attention(hidden_size=HQ * HEAD_DIM, num_heads=HQ, num_kv_heads=HKV, eps=1e-5).eval()
        attn.set_processor(Krea2MemoryEfficientAttnProcessor())
        with torch.no_grad():
            attn(torch.randn(1, 32, HQ * HEAD_DIM), attention_mask=None, image_rotary_emb=None)

        assert seen["enable_gqa"] is True
        assert seen["kv_heads"] == HKV, "the K/V heads were expanded despite the dispatcher serving them"
