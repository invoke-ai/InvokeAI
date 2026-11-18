"""The startup SDPA availability probe.

It exists to answer support questions from a log line instead of a probe script, so the properties
that matter are: it never breaks boot, it says nothing where the question does not apply, and what
it does say is specific enough to act on.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.nn.attention import SDPBackend

import invokeai.app.util.startup_utils as startup_utils
from invokeai.app.util.startup_utils import log_attention_backends, probe_attention_backends


@pytest.fixture(autouse=True)
def _forget_logged_devices():
    """The log is once per device per process, so a leftover entry would silently pass the next test."""
    startup_utils._probed_attention_devices.clear()
    yield
    startup_utils._probed_attention_devices.clear()


class TestProbe:
    @pytest.mark.parametrize("device_type", ["cpu", "mps"])
    def test_a_non_cuda_device_is_not_probed(self, device_type):
        # There are no fused SDPA backends to report, and can_use_* would raise.
        assert probe_attention_backends(torch.device(device_type)) is None

    def test_a_probe_failure_does_not_propagate(self):
        # A diagnostic must never be the reason the server does not start.
        with patch.object(torch, "empty", side_effect=RuntimeError("no CUDA driver")):
            assert probe_attention_backends(torch.device("cuda")) is None

    def test_a_failure_in_the_cleanup_does_not_propagate_either(self):
        """The cleanup used to sit in a `finally` on the same `try` as the handler, which does not
        cover it -- so a raise from `empty_cache()` escaped a function whose contract is that it
        never does."""
        with (
            patch.object(torch, "empty", return_value=MagicMock()),
            patch.object(torch.backends.cuda, "SDPAParams", return_value=MagicMock()),
            patch.object(torch.backends.cuda, "can_use_cudnn_attention", return_value=True),
            patch.object(torch.backends.cuda, "can_use_flash_attention", return_value=True),
            patch.object(torch.backends.cuda, "can_use_efficient_attention", return_value=True),
            patch.object(torch.cuda, "empty_cache", side_effect=RuntimeError("allocator is gone")),
        ):
            assert probe_attention_backends(torch.device("cuda")) is None

    def test_math_is_reported_as_always_available(self):
        # It is the unfused fallback, not a kernel that can be missing -- so it is never probed for.
        with (
            patch.object(torch, "empty", return_value=MagicMock()),
            patch.object(torch.backends.cuda, "SDPAParams", return_value=MagicMock()),
            patch.object(torch.backends.cuda, "can_use_cudnn_attention", return_value=False),
            patch.object(torch.backends.cuda, "can_use_flash_attention", return_value=False),
            patch.object(torch.backends.cuda, "can_use_efficient_attention", return_value=False),
            patch.object(torch.cuda, "empty_cache"),
        ):
            available = probe_attention_backends(torch.device("cuda"))
        assert available == {"cudnn": False, "flash": False, "efficient": False, "math": True}

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="the allocator is the thing under test")
    def test_the_probe_tensor_is_not_left_in_the_allocator(self):
        """Measured against the real allocator, because that is the claim.

        Patching `torch.empty` and `torch.cuda.empty_cache` and asserting the call happened tests
        choreography: it passes just as happily when the probe is still referenced and the call
        therefore frees nothing, which is what it did.
        """
        torch.cuda.empty_cache()
        before = torch.cuda.memory_reserved()

        probe_attention_backends(torch.device("cuda"))

        assert torch.cuda.memory_reserved() == before


class TestLogLine:
    def test_nothing_is_logged_where_the_question_does_not_apply(self, caplog):
        logger = logging.getLogger("test_probe_silent")
        with caplog.at_level(logging.INFO, logger=logger.name):
            log_attention_backends(logger, torch.device("cpu"))
        assert caplog.records == []

    def test_the_line_names_every_backend_and_its_answer(self, caplog, preserve_sdpa_priority_order):
        logger = logging.getLogger("test_probe_line")
        # An order neither torch nor InvokeAI uses, so the line can only have read it from torch.
        torch._C._set_sdp_priority_order(
            [
                int(SDPBackend.MATH),
                int(SDPBackend.CUDNN_ATTENTION),
                int(SDPBackend.FLASH_ATTENTION),
                int(SDPBackend.EFFICIENT_ATTENTION),
                int(SDPBackend.OVERRIDEABLE),
            ]
        )
        with patch(
            "invokeai.app.util.startup_utils.probe_attention_backends",
            return_value={"cudnn": True, "flash": False, "efficient": True, "math": True},
        ):
            with caplog.at_level(logging.INFO, logger=logger.name):
                log_attention_backends(logger, torch.device("cuda"))

        message = caplog.records[0].message
        assert "cudnn=yes" in message
        assert "flash=no" in message
        assert "efficient=yes" in message
        # The shape is part of the answer: availability depends on it, so a line without it would be
        # unactionable.
        assert "head_dim 128" in message
        # And the caveat that keeps it from being read as a dispatch table.
        assert "no mask" in message
        # Which of the available kernels a call gets is decided by the order, so the line states the one
        # in force.
        assert "priority MATH > CUDNN_ATTENTION > FLASH_ATTENTION > EFFICIENT_ATTENTION > OVERRIDEABLE" in message

    def test_a_backend_the_order_repeats_is_named_once_where_it_first_appears(
        self, caplog, preserve_sdpa_priority_order
    ):
        """torch 2.7's order has a fifth slot it never fills, so MATH appears twice. The repeat is never
        reached; naming it would read as a kernel tried twice."""
        logger = logging.getLogger("test_probe_repeat")
        torch._C._set_sdp_priority_order(
            [
                int(SDPBackend.FLASH_ATTENTION),
                int(SDPBackend.CUDNN_ATTENTION),
                int(SDPBackend.EFFICIENT_ATTENTION),
                int(SDPBackend.MATH),
                int(SDPBackend.MATH),
            ]
        )
        with patch(
            "invokeai.app.util.startup_utils.probe_attention_backends",
            return_value={"cudnn": True, "flash": False, "efficient": True, "math": True},
        ):
            with caplog.at_level(logging.INFO, logger=logger.name):
                log_attention_backends(logger, torch.device("cuda"))

        assert caplog.records[0].message.endswith(
            "priority FLASH_ATTENTION > CUDNN_ATTENTION > EFFICIENT_ATTENTION > MATH"
        )

    def test_it_is_logged_once_per_device(self, caplog):
        """The probe allocates, which creates that device's CUDA context. Repeating it per generation
        would pay that on a device the session may not even be using."""
        logger = logging.getLogger("test_probe_once")
        probe = MagicMock(return_value={"cudnn": True, "flash": False, "efficient": True, "math": True})
        with patch("invokeai.app.util.startup_utils.probe_attention_backends", probe):
            with caplog.at_level(logging.INFO, logger=logger.name):
                log_attention_backends(logger, torch.device("cuda:0"))
                log_attention_backends(logger, torch.device("cuda:0"))
                log_attention_backends(logger, torch.device("cuda:1"))

        assert probe.call_count == 2
        assert [r.message.split(" (fp16")[0] for r in caplog.records] == [
            "SDPA attention backends on cuda:0",
            "SDPA attention backends on cuda:1",
        ]

    def test_a_device_whose_probe_failed_is_not_probed_again(self):
        """Every claimed item asks. A probe allocates and ends with `empty_cache()`, and on a multi-GPU
        box that flush is deferred to a busy peer's step boundary -- retrying a failed probe would pay
        that on every item for a line that does not come."""
        logger = logging.getLogger("test_probe_retry")
        probe = MagicMock(return_value=None)
        with patch("invokeai.app.util.startup_utils.probe_attention_backends", probe):
            log_attention_backends(logger, torch.device("cuda:0"))
            log_attention_backends(logger, torch.device("cuda:0"))
        assert probe.call_count == 1
