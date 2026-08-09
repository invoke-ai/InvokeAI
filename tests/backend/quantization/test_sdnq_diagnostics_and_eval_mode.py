"""SDNQ hygiene: diagnostics must not cost anything when nobody is listening, and hand-built
modules must come back in inference mode.

Both are cheap to get wrong and invisible in normal use. The uint4 diagnostic ran full-tensor
reductions (and a `unique()` sort) on the first dequantization of every model and wrote to stdout,
bypassing the app's log level and handlers. The SDNQ loaders build their modules with
`init_empty_weights` + `load_state_dict` rather than `from_pretrained`, so nothing called `.eval()`
for them and any dropout / norm in the tree stayed in training mode during inference.
"""

import logging

import pytest
import torch

from invokeai.backend.model_manager.load.model_loaders.flux import _in_eval_mode
from invokeai.backend.quantization.sdnq import utils as sdnq_utils


@pytest.fixture(autouse=True)
def _reset_diagnostic_flag():
    sdnq_utils._uint4_diagnostic_done = False
    yield
    sdnq_utils._uint4_diagnostic_done = False


def _dequantize_uint4(out_features: int = 4, in_features: int = 64) -> torch.Tensor:
    """Run the uint4 per-group dequantization the diagnostic hangs off."""
    group_size = 32
    num_groups = in_features // group_size
    packed = torch.zeros(out_features, in_features // 2, dtype=torch.uint8)  # 2 nibbles per byte
    scale = torch.ones(out_features, num_groups, 1, dtype=torch.float32)
    zero_point = torch.zeros(out_features, num_groups, 1, dtype=torch.float32)
    return sdnq_utils.dequantize_uint4_per_group(
        packed, scale, zero_point, (out_features, in_features), group_size, torch.float32
    )


def test_the_uint4_diagnostic_writes_nothing_to_stdout(capsys) -> None:
    """stdout bypasses the app's log level, format and handlers — nothing may go there."""
    sdnq_utils.logger.setLevel(logging.DEBUG)

    _dequantize_uint4()

    assert capsys.readouterr().out == ""


def test_the_uint4_diagnostic_does_no_work_when_debug_logging_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """The figures cost more than the dequantization they describe, so they must be level-gated."""
    emitted: list[str] = []
    monkeypatch.setattr(sdnq_utils.logger, "isEnabledFor", lambda level: False)
    monkeypatch.setattr(sdnq_utils.logger, "debug", lambda *a, **k: emitted.append("debug"))

    _dequantize_uint4()

    assert emitted == []
    # The one-shot flag must stay unset, so the diagnostic is still available once debug is turned on.
    assert sdnq_utils._uint4_diagnostic_done is False


def test_the_uint4_diagnostic_runs_once_and_is_bounded_by_a_fixed_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    """Even with debug on, the diagnostic must not scale with the weight, and must not repeat."""
    assert sdnq_utils._DIAGNOSTIC_SAMPLE_SIZE <= 4096

    emitted: list[tuple] = []
    monkeypatch.setattr(sdnq_utils.logger, "isEnabledFor", lambda level: True)
    monkeypatch.setattr(sdnq_utils.logger, "debug", lambda *a, **k: emitted.append(a))

    big = _dequantize_uint4(out_features=64, in_features=1024)
    _dequantize_uint4(out_features=64, in_features=1024)

    assert len(emitted) == 1, "the diagnostic is one-shot"
    # Every reported element count is the bounded sample, never the tensor's own size.
    reported_counts = [arg for arg in emitted[0] if isinstance(arg, int)]
    assert big.numel() not in reported_counts
    assert sdnq_utils._DIAGNOSTIC_SAMPLE_SIZE in reported_counts


def test_sdnq_loaders_return_modules_in_eval_mode() -> None:
    """`_in_eval_mode` is what every SDNQ branch of the FLUX loader funnels through."""
    module = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.5))
    module.train()
    assert module.training

    returned = _in_eval_mode(module)

    assert returned is module
    assert all(not child.training for child in module.modules())


def test_in_eval_mode_passes_non_modules_through() -> None:
    """Tokenizers are not nn.Modules and must not be touched."""
    sentinel = object()

    assert _in_eval_mode(sentinel) is sentinel  # type: ignore[arg-type]
