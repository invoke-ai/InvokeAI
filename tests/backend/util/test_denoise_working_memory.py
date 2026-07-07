"""Tests for the architecture-scaled denoise working-memory estimator."""

import json
import logging
import types

import pytest
import torch
from accelerate import init_empty_weights
from diffusers import Flux2Transformer2DModel, QwenImageTransformer2DModel, ZImageTransformer2DModel

from invokeai.backend.util.denoise_working_memory import (
    ACTIVATION_MULTIPLIER,
    BASE_WORKING_MEMORY_BYTES,
    DEFAULT_ACTIVATION_WIDTH,
    GB,
    MB,
    DenoiseWorkingMemory,
    dtype_element_size,
    estimate_denoise_working_memory,
    estimate_denoise_working_memory_for_model,
    family_multiplier,
    model_activation_width,
)


def _model(**config_attrs):
    return types.SimpleNamespace(config=types.SimpleNamespace(**config_attrs))


def _estimate(model, lat_side: int, family: str, dtype: torch.dtype = torch.float16) -> DenoiseWorkingMemory:
    return estimate_denoise_working_memory_for_model(
        model=model,
        latent_height=lat_side,
        latent_width=lat_side,
        batch_size=1,
        inference_dtype=dtype,
        family=family,
    )


# --- core formula ---


def test_core_matches_explicit_formula():
    got = estimate_denoise_working_memory(
        latent_area=16384, activation_width=320, batch_size=1, element_size=2, multiplier=32
    )
    assert got == BASE_WORKING_MEMORY_BYTES + 32 * 16384 * 320 * 1 * 2


def test_core_scales_with_area_width_batch_and_element_size():
    base = estimate_denoise_working_memory(16384, 320, 1, 2, 32) - BASE_WORKING_MEMORY_BYTES
    assert estimate_denoise_working_memory(2 * 16384, 320, 1, 2, 32) - BASE_WORKING_MEMORY_BYTES == 2 * base
    assert estimate_denoise_working_memory(16384, 2 * 320, 1, 2, 32) - BASE_WORKING_MEMORY_BYTES == 2 * base
    assert estimate_denoise_working_memory(16384, 320, 2, 2, 32) - BASE_WORKING_MEMORY_BYTES == 2 * base
    assert estimate_denoise_working_memory(16384, 320, 1, 4, 32) - BASE_WORKING_MEMORY_BYTES == 2 * base


# --- width extraction from the model config ---


def test_model_activation_width_unet_uses_block_channels():
    assert model_activation_width(_model(block_out_channels=[320, 640, 1280, 1280])) == 320


def test_model_activation_width_transformer_hidden_dims():
    assert model_activation_width(_model(inner_dim=1536)) == 1536
    assert model_activation_width(_model(hidden_size=3072)) == 3072
    assert model_activation_width(_model(joint_attention_dim=2432)) == 2432
    assert model_activation_width(_model(dim=3840)) == 3840  # Z-Image-shaped config


def test_model_activation_width_from_heads_times_head_dim():
    assert model_activation_width(_model(num_attention_heads=24, attention_head_dim=128)) == 24 * 128


def test_model_activation_width_flux_params_hidden_size():
    flux_like = types.SimpleNamespace(config=None, params=types.SimpleNamespace(hidden_size=3072))
    assert model_activation_width(flux_like) == 3072


def test_model_activation_width_direct_attribute():
    """Custom models (Anima) expose the width directly, not under .config."""
    assert model_activation_width(types.SimpleNamespace(hidden_size=2048)) == 2048
    assert model_activation_width(types.SimpleNamespace(model_channels=1920)) == 1920


def test_model_activation_width_fallback():
    assert model_activation_width(object()) == DEFAULT_ACTIVATION_WIDTH


# --- width extraction from the REAL GGUF-shipped transformer classes ---
#
# The GGUF loaders build these exact diffusers classes under init_empty_weights() and swap in
# quantized weight tensors with load_state_dict(assign=True), so the structural config ints the
# width probe reads are untouched by quantization. If a diffusers attribute rename (or a new arch)
# breaks the probe, it silently falls back to DEFAULT_ACTIVATION_WIDTH: memory-safe (the cache
# floors the reserve) but the resolution-scaled partial-load headroom safeguard degenerates to the
# flat floor for exactly the models that need it. These tests make that regression loud.


@pytest.mark.parametrize(
    ("build", "expected_width"),
    [
        pytest.param(lambda: QwenImageTransformer2DModel(), 3584, id="qwen"),  # config.joint_attention_dim
        pytest.param(lambda: ZImageTransformer2DModel(), 3840, id="z_image"),  # config/direct-attr "dim"
        pytest.param(lambda: Flux2Transformer2DModel(), 15360, id="flux2_klein"),  # config.joint_attention_dim
    ],
)
def test_model_activation_width_resolves_for_gguf_transformers(build, expected_width):
    with init_empty_weights():
        model = build()  # meta device: structural config only, no weight allocation
    width = model_activation_width(model)
    assert width != DEFAULT_ACTIVATION_WIDTH, "width probe fell back to the default — headroom safeguard inert"
    assert width == expected_width


# --- the estimate object ---


def test_estimate_carries_its_inputs():
    """The estimate is self-describing so the DENOISE_MEM record needs no re-supplied state: it
    carries the family, the multiplier that produced it, and the dtype/batch it was computed for."""
    est = _estimate(_model(hidden_size=3072), 128, "flux2", dtype=torch.float32)
    assert est.family == "flux2"
    assert est.multiplier == ACTIVATION_MULTIPLIER["flux2"]
    assert est.element_size == 4
    assert est.batch_size == 1
    assert est.bytes == BASE_WORKING_MEMORY_BYTES + int(ACTIVATION_MULTIPLIER["flux2"] * 128 * 128 * 3072 * 1 * 4)


# --- calibration: the model-based estimate tracks the measured SDXL peaks ---


def test_unet_estimate_tracks_measured_sdxl_peaks():
    """SDXL (model_channels=320) measured ~332/737/1295 MB at 1024/1536/2048^2. The model-scaled
    estimate should be a conservative match: >= measured (never under-reserve), <= ~2x measured."""
    sdxl = _model(block_out_channels=[320, 640, 1280, 1280])
    measured_mb = {128: 332, 192: 737, 256: 1295}  # latent side -> measured MB
    for lat_side, m in measured_mb.items():
        est_mb = _estimate(sdxl, lat_side, "unet").bytes / MB
        assert m <= est_mb <= 2 * m, f"latent {lat_side}: est={est_mb:.0f}MB vs measured {m}MB"


def test_estimate_scales_up_for_wider_models():
    """A wider model (bigger hidden dim) must reserve more than a narrow one at the same resolution
    — this is the whole point of reading the width from the model instead of hardcoding."""
    narrow = _estimate(_model(hidden_size=1536), 128, "dit")
    wide = _estimate(_model(hidden_size=3072), 128, "dit")
    assert wide.bytes > narrow.bytes


def test_estimate_well_under_3gb_for_sdxl_normal_res():
    sdxl = _model(block_out_channels=[320, 640, 1280, 1280])
    for lat_side in (128, 192, 256):  # 1024/1536/2048 px
        assert _estimate(sdxl, lat_side, "unet").bytes < 3 * GB


def test_dit_estimate_covers_measured_flux_and_anima():
    """DiT multiplier is calibrated from FLUX.1 (implied ~5.3). The estimate must COVER (>=) each
    model's measured peak — FLUX (width 3072, ~583MB@1024^2) and Anima (width 2048, ~223MB) — never
    under-reserving, while staying a tight match to the FLUX anchor (<= ~1.6x)."""
    flux = types.SimpleNamespace(config=None, params=types.SimpleNamespace(hidden_size=3072))
    anima = types.SimpleNamespace(hidden_size=2048)  # direct attribute, no .config
    flux_mb = _estimate(flux, 128, "dit").bytes / MB
    anima_mb = _estimate(anima, 128, "dit").bytes / MB
    assert 583 <= flux_mb <= 1.6 * 583, f"flux est={flux_mb:.0f}MB vs measured 583"
    assert anima_mb >= 223, f"anima est={anima_mb:.0f}MB under measured 223"


def test_transformers_need_smaller_multiplier_than_unet():
    """Measured: transformers need far less working memory per unit width than the conv UNet
    (FLUX ~5.3, SDXL ~28), so the DiT multiplier is intentionally well below the UNet one."""
    assert ACTIVATION_MULTIPLIER["dit"] < ACTIVATION_MULTIPLIER["unet"]


def test_dtype_element_size():
    assert dtype_element_size(torch.float16) == 2
    assert dtype_element_size(torch.bfloat16) == 2
    assert dtype_element_size(torch.float32) == 4


# --- DENOISE_MEM telemetry: the logger the invocations actually pass ---


def test_probe_works_with_logger_interface():
    """Regression: every denoise node passes ``context.logger`` — a ``LoggerInterface``, NOT a
    ``logging.Logger`` — into ``measure()``, which calls ``logger.isEnabledFor(...)``. The wrapper
    must expose ``isEnabledFor`` or that call raises ``AttributeError`` on EVERY denoise. This
    reproduces the exact call site with a real LoggerInterface."""
    from invokeai.app.services.shared.invocation_context import LoggerInterface

    real = logging.getLogger("test_denoise_mem_regression")
    real.setLevel(logging.WARNING)  # DEBUG disabled

    ctx_logger = LoggerInterface.__new__(LoggerInterface)
    ctx_logger._services = types.SimpleNamespace(logger=real)

    # The wrapper must support the level check the probe relies on...
    assert ctx_logger.isEnabledFor(logging.DEBUG) is False
    assert ctx_logger.isEnabledFor(logging.WARNING) is True
    # ...and the exact calls the invocations make must not raise (inactive because DEBUG is off).
    probe = _estimate(_model(hidden_size=3072), 128, "flux2").measure(ctx_logger, pixel_height=1024, pixel_width=1024)
    assert probe.active is False
    probe.end()  # no-op, must not raise


class _StubLogger:
    def __init__(self) -> None:
        self.records: list[str] = []

    def isEnabledFor(self, level: int) -> bool:
        return True

    def debug(self, msg: str) -> None:
        self.records.append(msg)


def test_probe_emits_or_noops_without_raising():
    """With DEBUG enabled the probe must run cleanly: on CUDA it snapshots and emits one DENOISE_MEM
    record; without CUDA it no-ops. Never raises either way."""
    log = _StubLogger()
    est = _estimate(_model(hidden_size=3072), 128, "flux2")
    probe = est.measure(log, pixel_height=1024, pixel_width=1024)
    probe.end()
    if torch.cuda.is_available():
        assert len(log.records) == 1 and log.records[0].startswith("DENOISE_MEM ")
        payload = json.loads(log.records[0][len("DENOISE_MEM ") :])
        # The record must be self-describing for calibration: it carries the per-arch multiplier that
        # produced the estimate, so back-solving the multiplier needs no external state.
        assert payload["mult"] == ACTIVATION_MULTIPLIER["flux2"]
        assert payload["label"] == "flux2" and "measured_peak_mb" in payload
        assert "elapsed_ms" in payload  # denoise-loop wall time, for the on/off timing A/B
    else:
        assert probe.active is False
        assert log.records == []


def test_probe_label_override():
    """The label defaults to the family but can be overridden where one family covers several code
    paths (e.g. family "unet" emitting as "sdxl" vs "sdxl-legacy")."""
    log = _StubLogger()
    probe = _estimate(_model(hidden_size=1536), 64, "unet").measure(
        log, pixel_height=512, pixel_width=512, label="sdxl-legacy"
    )
    probe.end()
    if torch.cuda.is_available():
        payload = json.loads(log.records[0][len("DENOISE_MEM ") :])
        assert payload["label"] == "sdxl-legacy"
        assert payload["mult"] == ACTIVATION_MULTIPLIER["unet"]  # multiplier comes from the FAMILY


def test_probe_end_is_idempotent():
    log = _StubLogger()
    probe = _estimate(_model(hidden_size=3072), 128, "flux2").measure(log, pixel_height=1024, pixel_width=1024)
    probe.end()
    probe.end()
    assert len(log.records) <= 1


# --- per-architecture multipliers ---


def test_family_multiplier_per_arch():
    assert family_multiplier("flux2") == ACTIVATION_MULTIPLIER["flux2"] == 2.2
    assert family_multiplier("qwen") == 8.0
    assert family_multiplier("sd3") == 3.0
    assert family_multiplier("anima") == 3.5
    assert family_multiplier("z_image") == 4.5
    assert family_multiplier("unet") == ACTIVATION_MULTIPLIER["unet"] == 32
    # Unmeasured transformers fall back to the conservative "dit" default.
    assert family_multiplier("cogview4") == ACTIVATION_MULTIPLIER["dit"]
    assert family_multiplier("totally-unknown") == ACTIVATION_MULTIPLIER["dit"]


def test_per_arch_estimates_differ_on_same_model():
    """The family key selects the multiplier, so the same model reserves differently per arch:
    qwen (8.0) > z_image (4.5) > flux2 (2.2) at identical width/resolution."""
    m = _model(hidden_size=3072)
    qwen = _estimate(m, 128, "qwen").bytes
    z = _estimate(m, 128, "z_image").bytes
    flux2 = _estimate(m, 128, "flux2").bytes
    assert qwen > z > flux2
    # flux2 (2.2) reserves less than the generic "dit" default (6) — that's the per-arch win.
    assert flux2 < _estimate(m, 128, "dit").bytes
