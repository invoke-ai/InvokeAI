"""Tests for the architecture-scaled denoise working-memory estimator."""

import types

import torch

from invokeai.backend.util.denoise_working_memory import (
    ACTIVATION_MULTIPLIER,
    BASE_WORKING_MEMORY_BYTES,
    DEFAULT_ACTIVATION_WIDTH,
    ENFORCE_DIT_WORKING_MEMORY,
    ENFORCE_UNET_WORKING_MEMORY,
    GB,
    MB,
    begin_denoise_measure,
    dtype_element_size,
    end_denoise_measure,
    estimate_denoise_working_memory,
    estimate_denoise_working_memory_for_model,
    model_activation_width,
    resolve_denoise_working_mem_bytes,
)


def _model(**config_attrs):
    return types.SimpleNamespace(config=types.SimpleNamespace(**config_attrs))


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


def test_model_activation_width_from_heads_times_head_dim():
    assert model_activation_width(_model(num_attention_heads=24, attention_head_dim=128)) == 24 * 128


def test_model_activation_width_flux_params_hidden_size():
    flux_like = types.SimpleNamespace(config=None, params=types.SimpleNamespace(hidden_size=3072))
    assert model_activation_width(flux_like) == 3072


def test_model_activation_width_fallback():
    assert model_activation_width(object()) == DEFAULT_ACTIVATION_WIDTH


# --- calibration: the model-based estimate tracks the measured SDXL peaks ---


def test_unet_estimate_tracks_measured_sdxl_peaks():
    """SDXL (model_channels=320) measured ~332/737/1295 MB at 1024/1536/2048^2. The model-scaled
    estimate should be a conservative match: >= measured (never under-reserve), <= ~2x measured."""
    sdxl = _model(block_out_channels=[320, 640, 1280, 1280])
    measured_mb = {128: 332, 192: 737, 256: 1295}  # latent side -> measured MB
    for lat_side, m in measured_mb.items():
        est_mb = estimate_denoise_working_memory_for_model(sdxl, lat_side, lat_side, 1, 2, "unet") / MB
        assert m <= est_mb <= 2 * m, f"latent {lat_side}: est={est_mb:.0f}MB vs measured {m}MB"


def test_estimate_scales_up_for_wider_models():
    """A wider model (bigger hidden dim) must reserve more than a narrow one at the same resolution
    — this is the whole point of reading the width from the model instead of hardcoding."""
    narrow = estimate_denoise_working_memory_for_model(_model(hidden_size=1536), 128, 128, 1, 2, "dit")
    wide = estimate_denoise_working_memory_for_model(_model(hidden_size=3072), 128, 128, 1, 2, "dit")
    assert wide > narrow


def test_estimate_well_under_3gb_for_sdxl_normal_res():
    sdxl = _model(block_out_channels=[320, 640, 1280, 1280])
    for lat_side in (128, 192, 256):  # 1024/1536/2048 px
        assert estimate_denoise_working_memory_for_model(sdxl, lat_side, lat_side, 1, 2, "unet") < 3 * GB


# --- enforcement / dtype ---


def test_resolve_per_family_enforcement():
    assert resolve_denoise_working_mem_bytes(1234, "unet") == (1234 if ENFORCE_UNET_WORKING_MEMORY else None)
    assert resolve_denoise_working_mem_bytes(1234, "dit") == (1234 if ENFORCE_DIT_WORKING_MEMORY else None)


def test_dit_estimate_covers_measured_flux_and_anima():
    """DiT multiplier is calibrated from FLUX (clean ~5.3). The estimate must COVER (>=) each model's
    measured peak — FLUX (width 3072, ~583MB@1024^2) and Anima (width 2048, ~223MB) — never
    under-reserving, while staying a tight match to the FLUX anchor (<= ~1.6x)."""
    flux = types.SimpleNamespace(config=None, params=types.SimpleNamespace(hidden_size=3072))
    anima = types.SimpleNamespace(hidden_size=2048)  # direct attribute, no .config
    flux_mb = estimate_denoise_working_memory_for_model(flux, 128, 128, 1, 2, "dit") / MB
    anima_mb = estimate_denoise_working_memory_for_model(anima, 128, 128, 1, 2, "dit") / MB
    assert 583 <= flux_mb <= 1.6 * 583, f"flux est={flux_mb:.0f}MB vs measured 583"
    assert anima_mb >= 223, f"anima est={anima_mb:.0f}MB under measured 223"


def test_model_activation_width_direct_attribute():
    """Custom models (Anima) expose the width directly, not under .config."""
    assert model_activation_width(types.SimpleNamespace(hidden_size=2048)) == 2048
    assert model_activation_width(types.SimpleNamespace(model_channels=1920)) == 1920


def test_transformers_need_smaller_multiplier_than_unet():
    """Measured: transformers need far less working memory per unit width than the conv UNet
    (FLUX ~5.3, SDXL ~28), so the DiT multiplier is intentionally well below the UNet one."""
    assert ACTIVATION_MULTIPLIER["dit"] < ACTIVATION_MULTIPLIER["unet"]


def test_dtype_element_size():
    assert dtype_element_size(torch.float16) == 2
    assert dtype_element_size(torch.bfloat16) == 2
    assert dtype_element_size(torch.float32) == 4


# --- DENOISE_MEM telemetry: the logger the invocations actually pass ---


def test_begin_denoise_measure_works_with_logger_interface():
    """Regression: every denoise node passes ``context.logger`` — a ``LoggerInterface``, NOT a
    ``logging.Logger`` — to ``begin_denoise_measure``, which calls ``logger.isEnabledFor(...)``. The
    wrapper must expose ``isEnabledFor`` or that call raises ``AttributeError`` on EVERY denoise.
    This reproduces the exact call site with a real LoggerInterface."""
    import logging

    from invokeai.app.services.shared.invocation_context import LoggerInterface

    real = logging.getLogger("test_denoise_mem_regression")
    real.setLevel(logging.WARNING)  # DEBUG disabled

    ctx_logger = LoggerInterface.__new__(LoggerInterface)
    ctx_logger._services = types.SimpleNamespace(logger=real)

    # The wrapper must support the level check begin_denoise_measure relies on...
    assert ctx_logger.isEnabledFor(logging.DEBUG) is False
    assert ctx_logger.isEnabledFor(logging.WARNING) is True
    # ...and the exact call the invocations make must not raise (returns None because DEBUG is off).
    assert begin_denoise_measure(ctx_logger) is None


def test_begin_end_denoise_measure_emit_or_noop_without_raising():
    """With DEBUG enabled the pair must run cleanly: on CUDA it snapshots and emits one DENOISE_MEM
    record; without CUDA it no-ops. Never raises either way."""

    class _StubLogger:
        def __init__(self) -> None:
            self.records: list[str] = []

        def isEnabledFor(self, level: int) -> bool:
            return True

        def debug(self, msg: str) -> None:
            self.records.append(msg)

    log = _StubLogger()
    token = begin_denoise_measure(log)
    end_denoise_measure(
        token,
        log,
        label="flux2",
        estimate_bytes=512 * MB,
        pixel_height=1024,
        pixel_width=1024,
        batch_size=1,
        element_size=2,
    )
    if torch.cuda.is_available():
        assert token is not None
        assert len(log.records) == 1 and log.records[0].startswith("DENOISE_MEM ")
    else:
        assert token is None
        assert log.records == []
