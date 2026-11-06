"""A working-memory calibration service on a temporary root, with a faked GPU.

A device class and an environment can only be read from real hardware, so the service gets fixed ones: one class for
``cuda:0``, an environment whose hard facts are the settings passed in. Windows are built the way the probe reports
them, without measuring anything.
"""

import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import pytest
import torch

from invokeai.app.services.session_queue import session_queue_default
from invokeai.app.services.working_memory_calibration.working_memory_calibration_default import (
    DeviceMemory,
    SettingValue,
    WorkingMemoryCalibrationService,
)
from invokeai.app.services.working_memory_calibration.working_memory_calibration_plan import VaeDtypes
from invokeai.app.services.working_memory_calibration.working_memory_calibration_store import CalibrationStore
from invokeai.backend.working_memory.estimate import WorkingMemoryTerm
from invokeai.backend.working_memory.fingerprint import DeviceClass, Environment
from invokeai.backend.working_memory.probe import MeasuredWindow, Reason

GPU = torch.device("cuda:0")
GPU_CLASS = DeviceClass(vendor="nvidia", name="Test GPU 24GB", total_mib=24564, backend="cuda", arch="sm_89")
GPU_VAE_DTYPES = VaeDtypes(standalone_flux="bf16", z_image_pipeline="bf16", sdnq_quantized_vae=lambda config: False)
OTHER_CLASS = DeviceClass(vendor="amd", name="Test GPU 16GB", total_mib=16304, backend="hip", arch="gfx1200")


def fake_environment(device: torch.device, settings: Mapping[str, SettingValue]) -> Environment:
    hard = {"torch": "2.7", **{f"setting.{key}": str(value) for key, value in settings.items()}}
    return Environment(hard=hard, soft={"driver": "1.0", "os_build": "test"})


class HeldEnqueue:
    """Holds the next enqueue between its survey of the queue and its admission until released, as a busy worker
    thread would. Later enqueues pass."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        prepare = session_queue_default.prepare_values_to_insert

        def held(*args: Any, **kwargs: Any):
            monkeypatch.setattr(session_queue_default, "prepare_values_to_insert", prepare)
            self.entered.set()
            assert self.release.wait(timeout=10)
            return prepare(*args, **kwargs)

        monkeypatch.setattr(session_queue_default, "prepare_values_to_insert", held)


def calibration_service(
    root: Path,
    devices: Optional[Callable[[], list[torch.device]]] = None,
    device_class: DeviceClass = GPU_CLASS,
    environment_for: Callable[[torch.device, Mapping[str, SettingValue]], Environment] = fake_environment,
    untiled_decode_reservation: Optional[Callable[[str, int, int, str, torch.device], int]] = None,
    device_memory: Optional[Callable[[torch.device], DeviceMemory]] | Literal["whole card"] = "whole card",
    vae_dtypes: VaeDtypes = GPU_VAE_DTYPES,
) -> WorkingMemoryCalibrationService:
    """A service storing under `root`; not started (call `start(invoker)`, and `stop(invoker)` once it registered
    the user-fit provider estimates ask). What the cache can give a calibration's lock defaults to the whole card, with
    nothing outside torch; None asks the model cache of the invoker's model manager, as the service does. The VAE
    dtypes default to those of a GPU with bf16 under `precision: auto`."""
    extra = {} if untiled_decode_reservation is None else {"untiled_decode_reservation": untiled_decode_reservation}
    return WorkingMemoryCalibrationService(
        CalibrationStore(root / "working_memory"),
        generation_devices=devices or (lambda: [GPU]),
        device_class_for=lambda device: device_class,
        environment_for=environment_for,
        device_memory=(lambda device: DeviceMemory(device_class.total_mib * 2**20, 0))
        if device_memory == "whole card"
        else device_memory,
        vae_dtypes=lambda device, models_path: vae_dtypes,
        **extra,
    )


def measured_window(
    run_id: str,
    queue_item_id: int = 1,
    node_id: str = "decode",
    *,
    model_key: str = "vae-key",
    estimator: str = "vae_decode",
    version: int = 1,
    demand_bytes: int = 3 * 2**30,
    reasons: tuple[Reason, ...] = (),
) -> MeasuredWindow:
    valid = not any(reason != "partial_residency" for reason in reasons)
    return MeasuredWindow(
        run_id=run_id,
        queue_item_id=queue_item_id,
        node_id=node_id,
        node_type="test_calibration_node",
        model_key=model_key,
        device=str(GPU),
        estimator=estimator,
        version=version,
        features={"pixels": 1024 * 1024},
        terms=(WorkingMemoryTerm(name="activations", bytes=2.5 * 2**30, included=True),),
        reservation_bytes=int(2.5 * 2**30),
        source="data",
        reservation_applied_bytes=int(3.75 * 2**30),
        driver=None,
        reserved=None,
        allocated=None,
        demand_bytes=demand_bytes,
        method="torch-only",
        verified=False,
        resident_fraction=1.0,
        co_locked_models=[],
        segments=1,
        duration_s=0.5,
        valid=valid,
        fit_eligible=valid and "partial_residency" not in reasons,
        reasons=list(reasons),
    )
