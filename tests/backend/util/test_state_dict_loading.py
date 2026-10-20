"""Tests for the shared unexpected/missing key policy used by the single-file model loaders.

Issue #9437: loaders used to disagree about what an extra key in a checkpoint means, and the ones
that hard-failed turned exporter noise (`model_sampling.sigmas`, `pos_embedder.seq`, ...) into a
release-blocking crash. The policy pinned here is: extra keys are reported at DEBUG and ignored;
a *missing* required parameter is still an error.
"""

import logging

import pytest
import torch

from invokeai.backend.util.state_dict_loading import (
    MAX_REPORTED_KEYS,
    load_state_dict_ignoring_extras,
    log_unexpected_keys,
    reject_incomplete_load,
)

LOGGER_NAME = "invokeai.backend.util.state_dict_loading"


class _TinyNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)
        self.register_buffer("scale", torch.ones(4))


@pytest.fixture
def net() -> _TinyNet:
    return _TinyNet()


@pytest.fixture
def full_sd(net: _TinyNet) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in net.state_dict().items()}


class TestLogUnexpectedKeys:
    def test_reports_at_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
            log_unexpected_keys("tiny checkpoint", ["model_sampling.sigmas"])

        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.DEBUG
        assert "tiny checkpoint" in caplog.text
        assert "model_sampling.sigmas" in caplog.text

    def test_silent_above_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            log_unexpected_keys("tiny checkpoint", ["model_sampling.sigmas"])

        assert caplog.records == []

    def test_no_keys_logs_nothing(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
            log_unexpected_keys("tiny checkpoint", [])

        assert caplog.records == []

    def test_long_lists_are_truncated(self, caplog: pytest.LogCaptureFixture) -> None:
        """A bundled VAE contributes hundreds of extras; the log line must stay readable."""
        keys = [f"vae.decoder.block_{i:03d}.weight" for i in range(MAX_REPORTED_KEYS + 25)]
        with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
            log_unexpected_keys("tiny checkpoint", keys)

        assert f"ignoring {len(keys)} key(s)" in caplog.text
        assert "(+25 more)" in caplog.text
        assert keys[MAX_REPORTED_KEYS] not in caplog.text

    def test_non_string_keys_do_not_crash_the_report(self, caplog: pytest.LogCaptureFixture) -> None:
        """A `.pth` unpickles to whatever it contains, so a key need not be a string."""
        with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
            log_unexpected_keys("tiny checkpoint", [1, "real.key"])

        assert "real.key" in caplog.text


class TestLoadStateDictIgnoringExtras:
    def test_extra_key_loads_and_only_logs(
        self, net: _TinyNet, full_sd: dict[str, torch.Tensor], caplog: pytest.LogCaptureFixture
    ) -> None:
        """The Anima case: an exporter serialized a tensor the model has no slot for."""
        sd = full_sd | {"model_sampling.sigmas": torch.zeros(3)}

        with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
            missing = load_state_dict_ignoring_extras(net, sd, source="tiny checkpoint")

        assert missing == []
        assert "model_sampling.sigmas" in caplog.text
        torch.testing.assert_close(net.lin.weight, full_sd["lin.weight"])

    def test_missing_key_still_raises(self, net: _TinyNet, full_sd: dict[str, torch.Tensor]) -> None:
        """Completeness is the invariant worth enforcing — a required parameter never filled would
        otherwise blow up mid-inference instead of at load time."""
        del full_sd["lin.bias"]

        with pytest.raises(RuntimeError, match="lin.bias"):
            load_state_dict_ignoring_extras(net, full_sd, source="tiny checkpoint")

    def test_allowed_missing_is_tolerated(self, net: _TinyNet, full_sd: dict[str, torch.Tensor]) -> None:
        """Tied weights the caller re-shares after the load (T5's encoder.embed_tokens, Qwen3's
        lm_head) are legitimately absent from the file."""
        del full_sd["lin.bias"]

        missing = load_state_dict_ignoring_extras(net, full_sd, source="tiny checkpoint", allowed_missing={"lin.bias"})

        assert missing == ["lin.bias"]

    def test_allow_missing_defers_to_the_caller(self, net: _TinyNet, full_sd: dict[str, torch.Tensor]) -> None:
        """Callers that run their own completeness check (a sweep for tensors left on the meta
        device) opt out of the missing-key error entirely."""
        del full_sd["lin.bias"]

        missing = load_state_dict_ignoring_extras(net, full_sd, source="tiny checkpoint", allow_missing=True)

        assert missing == ["lin.bias"]

    def test_shape_mismatch_still_raises(self, net: _TinyNet, full_sd: dict[str, torch.Tensor]) -> None:
        """Dropping `strict=True` must not cost the size check that came with it."""
        full_sd["lin.weight"] = torch.zeros(8, 8)

        with pytest.raises(RuntimeError, match="size mismatch"):
            load_state_dict_ignoring_extras(net, full_sd, source="tiny checkpoint")

    def test_assign_is_passed_through(self, full_sd: dict[str, torch.Tensor]) -> None:
        """`assign=True` is how every meta-device loader materializes its parameters."""
        import accelerate

        with accelerate.init_empty_weights():
            model = _TinyNet()
        assert model.lin.weight.is_meta

        load_state_dict_ignoring_extras(model, full_sd, source="tiny checkpoint", assign=True)

        assert not any(t.is_meta for t in (*model.parameters(), *model.buffers()))


class TestRejectIncompleteLoad:
    """The completeness half of the policy, for loaders that cannot use the missing-key check —
    either because they legitimately tolerate some missing keys, or because the key list cannot see
    the problem at all."""

    def test_raises_when_a_parameter_is_left_on_meta(self) -> None:
        import accelerate

        with accelerate.init_empty_weights():
            model = _TinyNet()

        with pytest.raises(RuntimeError, match="tiny checkpoint is incomplete"):
            reject_incomplete_load(model, what="tiny checkpoint")

    def test_names_the_tensors_the_checkpoint_did_not_fill(self, full_sd: dict[str, torch.Tensor]) -> None:
        import accelerate

        with accelerate.init_empty_weights():
            model = _TinyNet()
        del full_sd["lin.bias"]
        model.load_state_dict(full_sd, strict=False, assign=True)

        with pytest.raises(RuntimeError, match="lin.bias") as exc_info:
            reject_incomplete_load(model, what="tiny checkpoint")
        assert "1 tensor(s)" in str(exc_info.value)

    def test_does_not_raise_for_a_fully_materialized_model(self, full_sd: dict[str, torch.Tensor]) -> None:
        import accelerate

        with accelerate.init_empty_weights():
            model = _TinyNet()
        model.load_state_dict(full_sd, strict=False, assign=True)

        reject_incomplete_load(model, what="tiny checkpoint")

    def test_a_persistent_buffer_left_on_meta_is_caught(self) -> None:
        """A parameters-only sweep would miss this; `strict=False` reports the buffer as missing but
        several callers deliberately tolerate missing keys."""
        model = _TinyNet()
        model.scale = torch.empty(4, device="meta")

        with pytest.raises(RuntimeError, match="scale"):
            reject_incomplete_load(model, what="tiny checkpoint")

    def test_a_non_persistent_buffer_is_not_a_false_positive(self) -> None:
        """The case `missing_keys` cannot see and this sweep must not invent: a buffer registered
        `persistent=False` (Anima has three) never appears in the state dict, and its module\'s
        constructor materializes it because `init_empty_weights` defaults to `include_buffers=False`.
        """
        import accelerate

        class _WithNonPersistentBuffer(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = torch.nn.Linear(4, 4)
                self.register_buffer("inv_freq", torch.ones(2), persistent=False)

        with accelerate.init_empty_weights():
            model = _WithNonPersistentBuffer()
        assert "inv_freq" not in model.state_dict()
        model.load_state_dict({k: torch.zeros(v.shape) for k, v in model.state_dict().items()}, assign=True)

        reject_incomplete_load(model, what="tiny checkpoint")
