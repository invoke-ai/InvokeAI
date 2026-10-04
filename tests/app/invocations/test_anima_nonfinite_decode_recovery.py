"""Tests for AnimaLatentsToImageInvocation._recover_nonfinite_decode's gating and retry logic.

The method is exercised directly (it reads nothing from ``self``), with a scripted stub VAE
and a plain-logger context, so the tests cover exactly the shipped decision tree: the
short-circuits for upstream/weight corruption, the retry ordering, which tensor is returned,
and the never-raise backstop.
"""

import logging
import math
from types import SimpleNamespace

import torch

import invokeai.app.invocations.vae.anima_latents_to_image as anima_l2i
from invokeai.app.invocations.vae.anima_latents_to_image import AnimaLatentsToImageInvocation


class _StubWanVae(torch.nn.Module):
    """A minimal module whose decode() returns scripted outputs in order."""

    def __init__(self, outputs: list[torch.Tensor]):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)  # gives the weight scanner something finite to scan
        self._outputs = list(outputs)
        self.decode_calls = 0

    def decode(self, latents: torch.Tensor, return_dict: bool = True):
        self.decode_calls += 1
        return (self._outputs.pop(0),)


def _ctx():
    return SimpleNamespace(logger=logging.getLogger("test_anima_nonfinite_decode_recovery"))


def _nan_image() -> torch.Tensor:
    t = torch.zeros(1, 3, 1, 4, 4)
    t[0, 0, 0, 0, 0] = math.nan
    return t


_LATENTS = torch.zeros(1, 16, 1, 2, 2)


def _recover(vae, decoded, latents_finite):
    # model_construct() skips pydantic validation; the method reads none of the input fields.
    invocation = AnimaLatentsToImageInvocation.model_construct()
    return invocation._recover_nonfinite_decode(_ctx(), vae, _LATENTS, decoded, latents_finite)


def test_nonfinite_latents_short_circuit_no_retry():
    vae = _StubWanVae([])
    decoded = _nan_image()
    out = _recover(vae, decoded, latents_finite=False)
    assert out is decoded
    assert vae.decode_calls == 0


def test_corrupt_weights_short_circuit_no_retry():
    vae = _StubWanVae([])
    vae.linear.weight.data[0, 0] = math.nan
    decoded = _nan_image()
    out = _recover(vae, decoded, latents_finite=True)
    assert out is decoded
    assert vae.decode_calls == 0


def test_plain_retry_heals(monkeypatch):
    flush_calls = {"n": 0}
    monkeypatch.setattr(anima_l2i, "force_real_empty_cache", lambda: flush_calls.__setitem__("n", flush_calls["n"] + 1))
    good = torch.zeros(1, 3, 1, 4, 4)
    vae = _StubWanVae([good])
    out = _recover(vae, _nan_image(), latents_finite=True)
    assert out is good
    assert vae.decode_calls == 1
    assert flush_calls["n"] == 0  # the forced flush must not run when a plain retry suffices


def test_forced_empty_cache_retry_heals(monkeypatch):
    flush_calls = {"n": 0}
    monkeypatch.setattr(anima_l2i, "force_real_empty_cache", lambda: flush_calls.__setitem__("n", flush_calls["n"] + 1))
    good = torch.zeros(1, 3, 1, 4, 4)
    vae = _StubWanVae([_nan_image(), good])
    out = _recover(vae, _nan_image(), latents_finite=True)
    assert out is good
    assert vae.decode_calls == 2
    assert flush_calls["n"] == 1


def test_unrecoverable_returns_last_retry(monkeypatch):
    monkeypatch.setattr(anima_l2i, "force_real_empty_cache", lambda: None)
    still_bad = _nan_image()
    vae = _StubWanVae([_nan_image(), still_bad])
    out = _recover(vae, _nan_image(), latents_finite=True)
    assert out is still_bad
    assert vae.decode_calls == 2


def test_retry_runtime_error_returns_original_decode(monkeypatch):
    monkeypatch.setattr(anima_l2i, "force_real_empty_cache", lambda: None)

    class _RaisingVae(_StubWanVae):
        def decode(self, latents, return_dict=True):
            self.decode_calls += 1
            raise RuntimeError("HIP error: out of memory")

    vae = _RaisingVae([])
    decoded = _nan_image()
    out = _recover(vae, decoded, latents_finite=True)
    assert out is decoded


def test_triage_failure_backstop_returns_original_decode(monkeypatch):
    """The never-raise contract: even the triage machinery blowing up (any exception type)
    must fall back to returning the corrupt decode, not fail the generation."""

    def boom(*args, **kwargs):
        raise TypeError("wrapper subclass without isfinite")

    monkeypatch.setattr(anima_l2i, "scan_module_for_nonfinite_weights", boom)
    vae = _StubWanVae([])
    decoded = _nan_image()
    out = _recover(vae, decoded, latents_finite=True)
    assert out is decoded
    assert vae.decode_calls == 0
