import math

import torch

from invokeai.backend.util.vae_decode_diagnostics import (
    allocator_state_summary,
    force_real_empty_cache,
    nonfinite_fraction,
    scan_module_for_nonfinite_weights,
)


class _TinyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.register_buffer("scale", torch.ones(4))
        self.register_buffer("counts", torch.zeros(4, dtype=torch.int64))


def test_scan_clean_module_reports_no_nonfinite():
    scan = scan_module_for_nonfinite_weights(_TinyModule(), "cuda")
    assert scan.clean
    assert scan.nonfinite_names == []
    # linear.weight, linear.bias, scale — the int64 buffer is skipped.
    assert scan.tensors_scanned == 3
    assert "finite" in scan.describe()


def test_scan_finds_and_names_nonfinite_tensor():
    module = _TinyModule()
    module.linear.weight.data[0, 0] = math.nan
    module.scale[1] = math.inf
    scan = scan_module_for_nonfinite_weights(module, "cuda")
    assert not scan.clean
    # Both corrupt tensors live on CPU, and the scan ran with a cuda compute-device type.
    assert scan.nonfinite_on_cpu == 2
    assert scan.nonfinite_on_compute == 0
    assert any(n.startswith("linear.weight@") for n in scan.nonfinite_names)
    assert any(n.startswith("scale@") for n in scan.nonfinite_names)
    assert "linear.weight" in scan.describe()


def test_scan_classifies_compute_device_by_type():
    module = _TinyModule()
    module.linear.weight.data[0, 0] = math.nan
    # Treat CPU as the compute device: the corrupt tensor must be counted on-compute.
    scan = scan_module_for_nonfinite_weights(module, "cpu")
    assert scan.nonfinite_on_compute == 1
    assert scan.nonfinite_on_cpu == 0


def test_scan_counts_tied_tensors_once():
    module = _TinyModule()
    module.tied = torch.nn.Parameter(module.linear.weight)  # same tensor under a second name
    scan = scan_module_for_nonfinite_weights(module, "cuda")
    assert scan.tensors_scanned == 3


def test_nonfinite_fraction():
    t = torch.zeros(10)
    assert nonfinite_fraction(t) == 0.0
    t[0] = math.nan
    t[1] = math.inf
    assert abs(nonfinite_fraction(t) - 0.2) < 1e-9
    assert nonfinite_fraction(torch.zeros(0)) == 0.0


def test_force_real_empty_cache_bypasses_wrapper(monkeypatch):
    calls = {"real": 0, "wrapper": 0}

    def real():
        calls["real"] += 1

    def wrapper():
        calls["wrapper"] += 1

    wrapper.__wrapped__ = real
    monkeypatch.setattr(torch.cuda, "empty_cache", wrapper)
    force_real_empty_cache()
    assert calls == {"real": 1, "wrapper": 0}


def test_force_real_empty_cache_without_wrapper(monkeypatch):
    calls = {"n": 0}

    def plain():
        calls["n"] += 1

    monkeypatch.setattr(torch.cuda, "empty_cache", plain)
    force_real_empty_cache()
    assert calls["n"] == 1


def test_allocator_state_summary_cpu_device_mentions_peer_state():
    summary = allocator_state_summary(torch.device("cpu"))
    assert "peer_generation_device_busy=" in summary


def test_force_real_empty_cache_bypasses_installed_peer_aware_wrapper(monkeypatch):
    """Integration: the bypass must reach the real entry point through the wrapper that
    install_peer_aware_empty_cache actually builds, in the exact scenario the recovery targets
    (a peer generation device mid-session, where the wrapper itself skips)."""
    from invokeai.backend.util.device_pool import GENERATION_DEVICE_POOL
    from invokeai.backend.util.devices import install_peer_aware_empty_cache

    calls = {"real": 0}

    def stub():
        calls["real"] += 1

    original = torch.cuda.empty_cache
    monkeypatch.setattr(torch.cuda, "empty_cache", stub)
    try:
        install_peer_aware_empty_cache()
        assert torch.cuda.empty_cache is not stub  # the wrapper is installed over the stub

        GENERATION_DEVICE_POOL.reset()
        GENERATION_DEVICE_POOL.set_generation_devices([torch.device("cuda:0"), torch.device("cuda:1")])
        GENERATION_DEVICE_POOL.acquire_session(torch.device("cuda:1"))
        try:
            torch.cuda.empty_cache()
            assert calls["real"] == 0  # the peer-aware wrapper skipped (peer busy)
            force_real_empty_cache()
            assert calls["real"] == 1  # the bypass reached the real entry point anyway
        finally:
            GENERATION_DEVICE_POOL.release_session(torch.device("cuda:1"))
            GENERATION_DEVICE_POOL.reset()
    finally:
        # monkeypatch restores the attribute it replaced; make certain the pristine original is
        # back regardless of ordering with the wrapper installation.
        torch.cuda.empty_cache = original
