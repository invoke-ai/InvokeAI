import asyncio
import datetime
import hashlib
import json
import os
import threading
import uuid
import weakref
from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Mapping
from contextlib import AbstractContextManager, ExitStack, asynccontextmanager
from dataclasses import dataclass
from functools import partial
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Final, NamedTuple, Optional

import torch
from pydantic import BaseModel

from invokeai.app.invocations.call_saved_workflow import CallSavedWorkflowInvocation
from invokeai.app.invocations.fields import WithBoard
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.services.events.events_common import (
    FastAPIEvent,
    FastAPIEventFunc,
    QueueItemsRetriedEvent,
    register_events,
)
from invokeai.app.services.model_records import UnknownModelException
from invokeai.app.services.session_queue.session_queue_common import (
    DEFAULT_QUEUE_ID,
    QUEUE_ITEM_STATUS,
    Batch,
    EnqueueIdempotencyConflictError,
    QueueItemLineage,
    RetriedQueueItem,
    SessionQueueItem,
    SessionQueueItemNotFoundError,
)
from invokeai.app.services.shared.graph import Graph
from invokeai.app.services.working_memory_calibration.working_memory_calibration_base import (
    WorkingMemoryCalibrationServiceBase,
)
from invokeai.app.services.working_memory_calibration.working_memory_calibration_common import (
    CALIBRATION_DESTINATION,
    CALIBRATION_ORIGIN_PREFIX,
    MODEL_CHANGED,
    STORE_SCHEMA_VERSION,
    CalibrationAttemptBusyError,
    CalibrationAttemptConflictError,
    CalibrationDeviceClass,
    CalibrationEnvironment,
    CalibrationErrorCode,
    CalibrationGraphError,
    CalibrationHoldBack,
    CalibrationModelChangedError,
    CalibrationModelIdentity,
    CalibrationPoint,
    CalibrationQueueFullError,
    CalibrationRun,
    CalibrationRunNotFoundError,
    CalibrationRunStatus,
    CalibrationStoreError,
    CalibrationStoreFile,
    CalibrationStoreUnusableError,
    CalibrationUnavailableError,
    ObservedItem,
    RemovedAttemptSteps,
    WorkingMemoryCalibrationDeviceClass,
    WorkingMemoryCalibrationEnvironment,
    WorkingMemoryCalibrationModelRun,
    WorkingMemoryCalibrationModelSummary,
    WorkingMemoryCalibrationPlan,
    WorkingMemoryCalibrationPointCount,
    WorkingMemoryCalibrationRequest,
    WorkingMemoryCalibrationResetResult,
    WorkingMemoryCalibrationRunCreated,
    WorkingMemoryCalibrationRunSummary,
    WorkingMemoryCalibrationStatus,
    calibration_origin,
    calibration_run_id,
)
from invokeai.app.services.working_memory_calibration.working_memory_calibration_fits import (
    HeldBack,
    measured_models,
    model_lines,
    recorded_outside_torch_bytes,
    run_models,
    user_fit_of,
)
from invokeai.app.services.working_memory_calibration.working_memory_calibration_plan import (
    VaeDtypes,
    build_plan,
    decode_lines_of,
)
from invokeai.app.services.working_memory_calibration.working_memory_calibration_store import (
    CalibrationStore,
    StoreRead,
)
from invokeai.app.util.misc import uuid_string
from invokeai.backend.model_manager.load.load_default import resolve_submodel_path
from invokeai.backend.model_manager.taxonomy import SubModelType
from invokeai.backend.quantization.sdnq.detection import is_sdnq_folder
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.vae_working_memory import dtype_name, estimate_untiled_vae_decode
from invokeai.backend.working_memory import fingerprint, user_fit
from invokeai.backend.working_memory.counters import Backend, backend_for, driver_method, method_verified, read_counters
from invokeai.backend.working_memory.fingerprint import DeviceClass, Environment
from invokeai.backend.working_memory.platform import Platform, platform_for, platform_for_environment
from invokeai.backend.working_memory.probe import MeasuredWindow
from invokeai.backend.working_memory.spec import EstimatorSpec, import_estimator_modules
from invokeai.backend.working_memory.user_fit import UserFit

if TYPE_CHECKING:
    from invokeai.app.services.invocation_services import InvocationServices
    from invokeai.app.services.invoker import Invoker

MEMORY_SETTINGS: Final = (
    "pytorch_cuda_alloc_conf",
    "attention_backend",
    "attention_type",
    "attention_slice_size",
    "sequential_guidance",
    "pid_memory_optimization",
    "wan_memory_optimization",
    "rocm_aotriton_experimental",
)
"""The InvokeAI settings that change how much working memory an operation takes; hard facts of a measurement's
environment."""

_RUN_ID_NAMESPACE: Final = uuid.UUID("3c1f6a52-8d0e-4b6f-9a57-2f0d5c8e41b3")
"""Derives a run id from an owner, a retry key and the request's fingerprint."""

_RUNNING: Final[frozenset[QUEUE_ITEM_STATUS]] = frozenset({"pending", "in_progress", "waiting"})

_RESET: Final = "was reset; a repeat of the request that started it cannot start it again"

_ATTEMPTS_LOCK: Final = "calibration-attempts"
"""This process's gate in front of the lock every attempt's creation takes (`_attempt_lock`)."""

SettingValue = str | int | float | bool | None


def _is_keyed(run_id: str) -> bool:
    """Whether a retry key named the run: its id is then derived (a UUID 5, see `_identify`), otherwise fresh
    (`uuid_string`, a UUID 4) and never repeated."""
    try:
        return uuid.UUID(run_id).version == 5
    except ValueError:
        return False


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _untiled_decode_reservation(estimator: str, height: int, width: int, dtype: str, device: torch.device) -> int:
    return estimate_untiled_vae_decode(estimator, height, width, dtype, device).reservation_bytes


@dataclass(frozen=True)
class DeviceMemory:
    """A calibration device's memory as the plan sizes for it, read now."""

    capacity_bytes: int
    """What the device's model cache can give a calibration's measured lock, the decode's working memory and its VAE
    together (`ModelCache.calibration_capacity`): with `reserve_vram_gb` and `max_cache_vram_gb` applied, and, on a
    device that shares system memory (MPS, an integrated XPU), what is allocated there now taken off, since nothing
    moves out of the way of the lock."""
    outside_torch_bytes: Optional[int]
    """What this process holds outside torch's allocator: the driver's figure for it less torch's reserve. None where
    the counters give no such pair: MPS reads no reserve beside the driver's figure, and XPU's driver figure is
    device-wide. On a device that shares system memory, the rest of what the process holds is already out of
    `capacity_bytes`, which counts the RAM the host has available."""


def _vae_dtypes(device: torch.device, models_path: Path) -> VaeDtypes:
    """What the VAE loaders choose on `device`, by the loaders' own rules, for models installed under `models_path`."""

    def sdnq_quantized_vae(config: Any) -> bool:
        # Where `FluxSDNQDiffusersModel._load_model` looks, with the path the loader makes absolute.
        pipeline = SimpleNamespace(path=models_path / config.path, submodels=getattr(config, "submodels", None))
        return is_sdnq_folder(resolve_submodel_path(pipeline, SubModelType.VAE, pipeline.path / SubModelType.VAE.value))

    return VaeDtypes(
        standalone_flux=dtype_name(TorchDevice.dtype_avoiding_float16(TorchDevice.choose_torch_dtype(device), device)),
        z_image_pipeline=dtype_name(TorchDevice.choose_bfloat16_safe_dtype(device)),
        sdnq_quantized_vae=sdnq_quantized_vae,
    )


def _measurement(backend: Optional[Backend]) -> tuple[Optional[str], Optional[bool]]:
    """The driver reader a window of a `backend` GPU here uses, and whether it is verified; both None for a device
    class no generation device here belongs to."""
    if backend is None:
        return None, None
    return driver_method(backend), method_verified(backend)


def _device_key(device: torch.device) -> str:
    """One name per device, whether or not it carries its index: the cache's devices may come without one."""
    if device.index is not None or device.type == "cpu":
        return str(device)
    index = torch.cuda.current_device() if device.type == "cuda" and torch.cuda.is_available() else 0
    return f"{device.type}:{index}"


FileSignature = Optional[tuple[int, int, int]]
"""A store file's mtime, size and inode (file index on Windows), None when it has none. Every write replaces the file,
so a change shows in at least one of them."""


@dataclass(frozen=True)
class _DeviceFit:
    """What `refresh_user_fits` found for one generation device, and what it found it from."""

    signature: FileSignature
    env_id: str
    platform: Platform
    fit: Optional[UserFit]
    file: Optional[CalibrationStoreFile]
    """The file the fit was made from, kept to re-fit it when only the runs held back changed."""
    held_back: HeldBack
    """The runs whose windows the fit left out, and why (`_held_back`)."""


def _add_env(file: CalibrationStoreFile, env: Environment) -> None:
    if all(known.id != env.id for known in file.envs):
        file.envs.append(CalibrationEnvironment.of(env))


def _prune_envs(file: CalibrationStoreFile) -> None:
    used = {run.env_id for run in file.runs} | {point.env_id for point in file.points}
    file.envs = [env for env in file.envs if env.id in used]


def _add_points(
    file: CalibrationStoreFile,
    *,
    run_id: str,
    env: Environment,
    points: list[CalibrationPoint],
    epochs: Mapping[str, int],
) -> tuple[bool, int]:
    """Attach `points` to their run, except those attributed (`measured_models`, as a model reset removes them) to a
    model whose reset epoch moved past `epochs`: a reset removed the model's points after the points were measured.
    How many it attached; none when the file lacks the run."""
    if all(run.run_id != run_id for run in file.runs):
        return False, 0
    current = file.model_reset_epochs
    models = run_models(file)
    kept = [
        point
        for point in points
        if all(current.get(model, 0) <= epochs.get(model, 0) for model in measured_models(point, models))
    ]
    if not kept:
        return False, 0
    _add_env(file, env)
    file.points.extend(kept)
    return True, len(kept)


AttemptKey = tuple[str, str]
"""An attempt: the owner of its runs and its `attempt_id`. Two owners' attempts never share runs, whatever their ids."""


def _attempt_key(run: CalibrationRun) -> Optional[AttemptKey]:
    return (run.user_id, run.attempt_id) if run.attempt_id is not None else None


def _removed_steps(file: CalibrationStoreFile) -> dict[AttemptKey, RemovedAttemptSteps]:
    return {(record.user_id, record.attempt_id): record for record in file.removed_attempt_steps}


def _queued_runs(runs: list[CalibrationRun], removed: Optional[RemovedAttemptSteps]) -> tuple[int, int]:
    """How many runs the attempt of `runs` queued, those resets removed (`removed`) included, and how many it planned:
    the `attempt_steps` its runs sent, which the calibrate route keeps the same across them. A run without an attempt
    plans itself. Runs stored before attempt_steps was sent come only from development builds of this calibration,
    which never shipped; their attempt is judged on the runs it recorded, as it was then, rather than making their
    store file unusable."""
    queued = len(runs) + (removed.removed if removed is not None else 0)
    steps = {run.attempt_steps for run in runs} | {removed.attempt_steps if removed is not None else None}
    return queued, max((step for step in steps if step is not None), default=queued)


def _note_removed_steps(
    file: CalibrationStoreFile, removed: list[CalibrationRun], reasons: Mapping[str, Optional[CalibrationHoldBack]]
) -> None:
    """Note per attempt what a reset removed of it (`removed_attempt_steps`), once `file.runs` holds the runs left,
    and drop the notes nothing references any more.

    A removed run counts as queued. Its attempt is marked aborted when it was running (a later step of it may still
    come, with its earlier ones gone) or aborted (by a canceled item of the removed run, say), as `reasons`
    (`_hold_reasons`) had it when the reset took the file's lock; a run unknown to them is taken for running. A note
    stays while a stored run of its attempt is left, or while a later step of it may still come: this reset found the
    attempt running."""
    notes = _removed_steps(file)
    running: set[AttemptKey] = set()
    for run in removed:
        if (key := _attempt_key(run)) is None:
            continue
        reason = reasons.get(run.run_id, "running")
        note = notes.get(key)
        notes[key] = RemovedAttemptSteps(
            user_id=key[0],
            attempt_id=key[1],
            removed=(note.removed if note is not None else 0) + 1,
            aborted=(note is not None and note.aborted) or reason is not None,
            attempt_steps=note.attempt_steps
            if note is not None and note.attempt_steps is not None
            else run.attempt_steps,
        )
        if reason == "running":
            running.add(key)
    referenced = {key for run in file.runs if (key := _attempt_key(run)) is not None} | running
    file.removed_attempt_steps = [note for key, note in notes.items() if key in referenced]


def _lineage_outcomes(
    run: CalibrationRun, live: Mapping[int, QueueItemLineage]
) -> tuple[list[int], dict[int, QUEUE_ITEM_STATUS]]:
    """Every item of a run known, and the status of each item that stands for its lineage: the latest retry of an
    item, or the item itself. `live` as the queue has them now, the run's own observations for items the queue no
    longer has. An item the queue no longer has counts as canceled unless the runner saw it finish. A retry stands in
    for the item it retried, also before the runner started it."""
    item_ids = sorted(set(run.queue_item_ids) | set(run.observed) | set(live))

    def status_of(item_id: int) -> QUEUE_ITEM_STATUS:
        if item_id in live:
            return live[item_id].status
        observed = run.observed.get(item_id)
        # Gone from the queue while the runner last saw it running (a crash, a cleared queue): it will not finish.
        if observed is None or observed.status in _RUNNING:
            return "canceled"
        return observed.status

    def retried_from(item_id: int) -> Optional[int]:
        if item_id in live:
            return live[item_id].retried_from_item_id
        observed = run.observed.get(item_id)
        return observed.retried_from_item_id if observed is not None else None

    latest: dict[int, int] = {}
    for item_id in item_ids:
        source = retried_from(item_id)
        lineage = item_id if source is None else source
        latest[lineage] = max(latest.get(lineage, item_id), item_id)
    # A retry names the item it came from, which may be known by nothing else any more.
    item_ids = sorted(set(item_ids) | set(latest))
    return item_ids, {item_id: status_of(item_id) for item_id in sorted(latest.values())}


def _record_retry(run_id: str, item: SessionQueueItem, file: CalibrationStoreFile) -> tuple[bool, None]:
    """A store change: record the retry `item` on the run `run_id`, pending, unless the runner already observed it
    (admitted, or even settled it before the announcement came)."""
    run = next((run for run in file.runs if run.run_id == run_id), None)
    if run is None or item.item_id in run.observed:
        return False, None
    run.observed[item.item_id] = ObservedItem(
        status="pending", retried_from_item_id=item.retried_from_item_id, observed_at=_now()
    )
    return True, None


def _interrupts(run: CalibrationRun, item_id: int, status: QUEUE_ITEM_STATUS) -> bool:
    """Whether the item that stands for a lineage of `run` interrupted its attempt: canceled, or failed at admission
    because a model changed after the calibration was planned."""
    if status == "canceled":
        return True
    observed = run.observed.get(item_id)
    return status == "failed" and observed is not None and observed.error_code == MODEL_CHANGED


def _retry_listener(service: "WorkingMemoryCalibrationService") -> FastAPIEventFunc[QueueItemsRetriedEvent]:
    """The event bus's handler for `service`'s `_on_queue_items_retried`. The bus keeps every handler for the
    process's life; holding the service weakly lets a stopped one go."""
    reference = weakref.ref(service)

    async def on_queue_items_retried(event: FastAPIEvent[QueueItemsRetriedEvent]) -> None:
        listener = reference()
        if listener is not None:
            await listener._on_queue_items_retried(event)

    return on_queue_items_retried


class RunStatus(NamedTuple):
    status: CalibrationRunStatus
    error: Optional[str]
    error_code: Optional[CalibrationErrorCode]
    item_ids: list[int]


def derive_run_status(run: CalibrationRun, live: Mapping[int, QueueItemLineage], valid_point_count: int) -> RunStatus:
    """A run's status, error (with its code, where the runner gave one) and items, from its items
    (`_lineage_outcomes`). A failed item decides over a canceled one. A run that ended without completing is partial
    only with valid points: an invalid one (a canceled or failed operation, a spill) measured nothing usable."""
    item_ids, statuses = _lineage_outcomes(run, live)
    if not statuses:
        return RunStatus("canceled", None, None, item_ids)
    if any(status in _RUNNING for status in statuses.values()):
        return RunStatus("running", None, None, item_ids)
    failed = [item_id for item_id, status in statuses.items() if status == "failed"]
    if failed:
        observed = run.observed.get(failed[0])
        error = observed.error_message if observed is not None and observed.error_message else None
        code = observed.error_code if observed is not None else None
        return RunStatus("failed", error or f"Queue item {failed[0]} failed", code, item_ids)
    if all(status == "completed" for status in statuses.values()):
        return RunStatus("completed", None, None, item_ids)
    return RunStatus("partial" if valid_point_count > 0 else "canceled", None, None, item_ids)


def _model_identity(config: Any) -> CalibrationModelIdentity:
    def name(field: object) -> Optional[str]:
        return None if field is None else str(getattr(field, "value", field))

    return CalibrationModelIdentity(
        base=name(getattr(config, "base", None)),
        type=name(getattr(config, "type", None)),
        format=name(getattr(config, "format", None)),
        variant=name(getattr(config, "variant", None)),
    )


def _identity_changes(run: CalibrationRun, model_hash: Optional[str], identity: CalibrationModelIdentity) -> list[str]:
    """What differs between the model `run` recorded at its creation and one of `model_hash` and `identity`: the hash,
    and, for a run that recorded its identity (runs stored before it compare the hash alone), base, type, format and
    variant."""
    recorded = run.model_identity.model_dump() if run.model_identity is not None else {}
    current = identity.model_dump()
    facts = [fact for fact, value in recorded.items() if current[fact] != value]
    return (["hash"] if model_hash != run.model_hash else []) + facts


def _model_identifiers(graph: Graph) -> Iterator[ModelIdentifierField]:
    """The models a graph's nodes name: their model fields, also inside a field's model or list (a LoRA collection's
    `LoRAField.lora`)."""

    def within(value: object) -> Iterator[ModelIdentifierField]:
        if isinstance(value, ModelIdentifierField):
            yield value
        elif isinstance(value, BaseModel):
            for _, field in value:
                yield from within(field)
        elif isinstance(value, (list, tuple)):
            for entry in value:
                yield from within(entry)

    for node in graph.nodes.values():
        for _, value in node:
            yield from within(value)


class _Enqueue(NamedTuple):
    """What a creation enqueues: the batch, the run's device class, whether this request recorded the run, and
    whether it marks the run's enqueue, and so may clear the mark again when nothing was queued."""

    batch: Batch
    device_class_id: str
    created: bool
    marks: bool


class WorkingMemoryCalibrationService(WorkingMemoryCalibrationServiceBase):
    """Calibration runs and their store.

    `generation_devices`, `device_class_for`, `environment_for`, `untiled_decode_reservation`, `device_memory` and
    `vae_dtypes` default to the machine's own; tests pass their own because a device class, its memory and the dtypes
    it supports can only be read from real hardware.
    """

    def __init__(
        self,
        store: CalibrationStore,
        *,
        generation_devices: Optional[Callable[[], list[torch.device]]] = None,
        device_class_for: Callable[[torch.device], DeviceClass] = fingerprint.device_class_for,
        environment_for: Callable[
            [torch.device, Mapping[str, SettingValue]], Environment
        ] = fingerprint.environment_for,
        untiled_decode_reservation: Callable[[str, int, int, str, torch.device], int] = _untiled_decode_reservation,
        device_memory: Optional[Callable[[torch.device], DeviceMemory]] = None,
        vae_dtypes: Callable[[torch.device, Path], VaeDtypes] = _vae_dtypes,
    ) -> None:
        self._store = store
        self._generation_devices = generation_devices
        self._device_class_for = device_class_for
        self._environment_for = environment_for
        self._untiled_decode_reservation = untiled_decode_reservation
        self._device_memory = device_memory or self._cache_memory
        self._vae_dtypes = vae_dtypes
        # A card's class cannot change while the process runs; reading it touches the driver.
        self._device_classes: dict[str, DeviceClass] = {}
        # Which file holds a run, so admitting an item and recording its windows do not scan every file.
        self._run_files: dict[str, str] = {}
        # Creations in flight, referenced here so that one whose request was canceled still runs to its end.
        self._creations: set[asyncio.Task[WorkingMemoryCalibrationRunCreated]] = set()
        # Per run id while a creation of this process uses it, in front of the file lock (see `_creation_lock`).
        self._run_gates: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()
        self._invoker: Optional["Invoker"] = None
        # The user fit per generation device: replaced whole by `refresh_user_fits`, read by every estimate.
        self._fits: Mapping[str, _DeviceFit] = {}
        self._fits_lock = threading.Lock()
        # The status of each calibration item `_canceled_runs` found finished in the queue: item ids are never reused,
        # and a finished item never changes, so an entry holds for the process's lifetime.
        self._item_outcomes: dict[int, QUEUE_ITEM_STATUS] = {}
        # Whether a started service records the retries the queue announces (`_on_queue_items_retried`).
        self._listening = False
        # One object, so that `stop` unregisters exactly what `start` registered.
        self._provider: user_fit.UserFitProvider = self._user_fit_for

    def start(self, invoker: "Invoker") -> None:
        self._invoker = invoker
        user_fit.set_user_fit_provider(self._provider)
        self._listening = True
        register_events(QueueItemsRetriedEvent, _retry_listener(self))

    def stop(self, invoker: "Invoker") -> None:
        # The listener stays registered (the event bus keeps no unregister); it does nothing once stopped, and holds
        # the service only weakly.
        self._listening = False
        user_fit.clear_user_fit_provider(self._provider)

    @property
    def _services(self) -> "InvocationServices":
        assert self._invoker is not None, "The calibration service has not been started"
        return self._invoker.services

    # region Devices and environments

    def _devices(self) -> list[torch.device]:
        if self._generation_devices is not None:
            return list(self._generation_devices())
        configured = TorchDevice.get_generation_devices(self._services.configuration.generation_devices)
        return configured or [TorchDevice.choose_torch_device()]

    def _calibration_device(self) -> torch.device:
        devices = self._devices()
        if len(devices) > 1:
            raise CalibrationUnavailableError(
                f"Calibration runs on a single generation device, and this machine generates on {len(devices)} "
                f"({', '.join(str(device) for device in devices)}). Set generation_devices to one GPU to calibrate it."
            )
        if not devices or backend_for(devices[0]) is None:
            raise CalibrationUnavailableError(
                f"Calibration measures GPU memory, and this machine generates on {devices[0] if devices else 'no device'}."
            )
        return devices[0]

    def _device_class(self, device: torch.device) -> DeviceClass:
        key = str(device)
        if key not in self._device_classes:
            self._device_classes[key] = self._device_class_for(device)
        return self._device_classes[key]

    def _environment(self, device: torch.device) -> Environment:
        config = self._services.configuration
        settings: dict[str, SettingValue] = {name: getattr(config, name) for name in MEMORY_SETTINGS}
        return self._environment_for(device, settings)

    def _current_device_class_id(self, device_class_id: Optional[str]) -> str:
        if device_class_id is not None:
            return device_class_id
        return self._device_class(self._calibration_device()).id

    # endregion

    # region Runs

    def _find_run(self, run_id: str) -> Optional[tuple[str, CalibrationRun]]:
        cached = self._run_files.get(run_id)
        candidates = [cached] if cached is not None else []
        candidates += [
            device_class_id for device_class_id in self._store.device_class_ids() if device_class_id != cached
        ]
        for device_class_id in candidates:
            read = self._store.read(device_class_id)
            if read is None or read.file is None:
                continue
            for run in read.file.runs:
                if run.run_id == run_id:
                    self._run_files[run_id] = device_class_id
                    return device_class_id, run
        self._run_files.pop(run_id, None)
        return None

    @staticmethod
    def _identify(model_key: str, request: WorkingMemoryCalibrationRequest, user_id: str) -> str:
        """Check the request's graph, turn its node cache off and make every node's output an intermediate on no
        board, then return the run's id, which also becomes the graph's id.

        Without a retry key the id is fresh. With one it is derived from the owner, the key, the request's
        fingerprint (the model and the batch the server builds: graph and batch data) and the client's attempt, if it
        names one. A repeat of the request finds its run by that id, and another request reusing the key, another
        attempt included, gets another id. The id is part of the batch's origin, so the queue's receipt for the key
        refuses that other request. The client's graph id is not part of the request: a graph sent without one gets a
        fresh id each time, so a repeat could never match.
        """
        graph = request.graph
        for node in graph.nodes.values():
            if isinstance(node, CallSavedWorkflowInvocation):
                raise CalibrationGraphError(
                    f"Node {node.id} calls a saved workflow. A calibration graph cannot: the called workflow would "
                    "run as separate queue items, outside the measured node."
                )
        for datum in chain.from_iterable(request.data or []):
            if datum.field_name == "use_cache":
                raise CalibrationGraphError(
                    f"Batch data sets use_cache on node {datum.node_path}; calibration runs every node uncached."
                )
            # A board cannot come as batch data: batch items are strings, numbers, images and videos.
            if datum.field_name == "is_intermediate":
                raise CalibrationGraphError(
                    f"Batch data sets is_intermediate on node {datum.node_path}; a calibration's images are "
                    "intermediates."
                )
        for node in graph.nodes.values():
            # A cached output would skip the very operation the run measures.
            node.use_cache = False
            # Its images are measurement by-products: never in the gallery or on a board; clearing intermediates
            # removes them. A node saves with its own flag and its `WithBoard` board.
            node.is_intermediate = True
            if isinstance(node, WithBoard):
                node.board = None
        if request.idempotency_key is None:
            run_id = uuid_string()
        else:
            fingerprint = {
                "model_key": model_key,
                "graph": graph.model_dump(mode="json", exclude={"id"}),
                "data": None
                if request.data is None
                else [[datum.model_dump(mode="json") for datum in group] for group in request.data],
            }
            digest = hashlib.sha256(
                json.dumps(fingerprint, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
            ).hexdigest()
            # Appended only with an attempt: a request without one keeps the id it had before attempts existed.
            attempt = f"\x00{request.attempt_id}" if request.attempt_id is not None else ""
            run_id = str(uuid.uuid5(_RUN_ID_NAMESPACE, f"{user_id}\x00{request.idempotency_key}\x00{digest}{attempt}"))
        # The queue's receipt hashes the batch, graph id included: a repeat must build the same one.
        graph.id = run_id
        return run_id

    @asynccontextmanager
    async def _creation_lock(self, run_id: str) -> AsyncIterator[None]:
        """Hold the run's creation lock, which every process serving the root takes (`CalibrationStore.creation_lock`):
        a repeat of the request waits for the attempt in flight, in this process or another, and then finds what it
        did. Only a retry key's run id can be contended; a fresh one is not locked.

        In front of it, the repeats of one process wait on the run's gate, on the event loop: only one of them at a
        time takes a worker thread to wait for the file lock. A burst of repeats would otherwise fill the default
        executor, stalling the holder's own worker-thread steps and a shutdown that waits for the executor."""
        if not _is_keyed(run_id):
            yield
            return
        async with self._file_lock(run_id, partial(self._store.creation_lock, run_id)):
            yield

    @asynccontextmanager
    async def _attempt_lock(self, request: WorkingMemoryCalibrationRequest) -> AsyncIterator[None]:
        """For a run of an attempt, hold the lock every attempt's creation takes, from the run's record through its
        enqueue to the record of its items, in every process serving the root: the next attempt's creation finds the
        items of the last one in the queue, and refuses while they run (`CalibrationAttemptBusyError`). The runs of one
        attempt are posted one after the other, so they barely wait. A run without an attempt takes no lock."""
        if request.attempt_id is None:
            yield
            return
        async with self._file_lock(_ATTEMPTS_LOCK, self._store.attempts_lock):
            yield

    @asynccontextmanager
    async def _file_lock(self, name: str, os_lock: Callable[[], AbstractContextManager[None]]) -> AsyncIterator[None]:
        """Hold the store's OS lock `os_lock`, behind this process's gate `name` for it (`_creation_lock`)."""
        gate = self._run_gates.get(name)
        if gate is None:
            gate = self._run_gates[name] = asyncio.Lock()
        async with gate:
            held = ExitStack()
            # A file lock: taken in a worker thread, which waits for another holder without blocking the loop.
            acquiring = asyncio.ensure_future(asyncio.to_thread(held.enter_context, os_lock()))
            try:
                await asyncio.shield(acquiring)
            except asyncio.CancelledError:
                # The worker may still take the lock: give it back once it has.
                acquiring.add_done_callback(lambda _: held.close())
                raise
            try:
                yield
            finally:
                held.close()

    async def create_run(
        self, model_key: str, request: WorkingMemoryCalibrationRequest, user_id: str
    ) -> WorkingMemoryCalibrationRunCreated:
        # The enqueue runs in a worker thread and can commit after the request was canceled. So the creation runs to
        # its end whatever happens to the request: the run keeps the items that were queued, or is removed. Only the
        # server's shutdown cancels the creation as well; the run stays as far as it got, for a keyed repeat to finish.
        creation = asyncio.create_task(self._create_run(model_key, request, user_id))
        self._creations.add(creation)
        creation.add_done_callback(self._creations.discard)
        try:
            return await asyncio.shield(creation)
        except asyncio.CancelledError:
            creation.add_done_callback(self._log_unawaited_failure)
            raise

    def _log_unawaited_failure(self, creation: "asyncio.Task[WorkingMemoryCalibrationRunCreated]") -> None:
        """Nobody awaits the creation of a canceled request: its failure would go unseen."""
        if not creation.cancelled() and (error := creation.exception()) is not None:
            self._services.logger.warning(
                "The creation of a calibration run whose request was canceled failed", exc_info=error
            )

    async def _create_run(
        self, model_key: str, request: WorkingMemoryCalibrationRequest, user_id: str
    ) -> WorkingMemoryCalibrationRunCreated:
        run_id = await asyncio.to_thread(self._identify, model_key, request, user_id)
        # Recording the run, enqueuing its batch and recording its items or removing it: one step for each run id, and
        # for every attempt, in every process serving the root.
        async with self._attempt_lock(request), self._creation_lock(run_id):
            prepared = await asyncio.to_thread(self._prepare_run, model_key, request, user_id, run_id)
            if isinstance(prepared, WorkingMemoryCalibrationRunCreated):
                return prepared
            batch, device_class_id, created, marks = prepared
            if marks and not created:
                # Before the enqueue: once it may have committed, a repeat that finds no items must not enqueue again.
                # A run this request recorded was marked as it was recorded.
                await asyncio.to_thread(self._set_enqueue_attempted, device_class_id, run_id, True)
            try:
                result = await self._services.session_queue.enqueue_batch(
                    queue_id=DEFAULT_QUEUE_ID, batch=batch, prepend=False, user_id=user_id, all_or_none=True
                )
            except Exception:
                # The queue announces a batch after committing it, so an enqueue can raise once its items are queued.
                # Items of an earlier attempt that a reset left behind carry the run's origin too, and answer a repeat
                # of the request as the key's receipt would.
                try:
                    queued: Optional[list[int]] = await asyncio.to_thread(self._queued_item_ids, run_id)
                except Exception:
                    self._services.logger.warning(
                        f"Could not tell whether calibration run {run_id} has queue items; it is kept", exc_info=True
                    )
                    queued = None
                if queued is None:
                    # The enqueue's own error answers the request.
                    raise
                if not queued:
                    await asyncio.to_thread(self._enqueued_nothing, device_class_id, run_id, created, marks)
                    raise
                self._services.logger.warning(
                    f"Enqueuing calibration run {run_id} raised, and the queue holds its items {queued}",
                    exc_info=True,
                )
                item_ids = queued
            else:
                if result.enqueued == 0:
                    await asyncio.to_thread(self._enqueued_nothing, device_class_id, run_id, created, marks)
                    # A partial run would measure some of the batch's cases, so the queue takes all of them or none.
                    raise CalibrationQueueFullError(
                        f"The queue is full: it has no room for all {result.requested} items of the calibration run, "
                        "so the run was not started."
                    )
                item_ids = list(result.item_ids)
            try:
                await asyncio.to_thread(self._record_items, device_class_id, run_id, item_ids)
            except CalibrationStoreError as error:
                # An unusable file fails the run's items at admission, as a removed run (not caught here) does: an error
                # with a key or without. With a key, a repeat records the items, from the key's receipt or, once the
                # queue has collected that, from their origin. Without one the request cannot be repeated, only sent
                # again, which would queue the batch twice: a passing store failure answers with the items.
                if isinstance(error, CalibrationStoreUnusableError) or request.idempotency_key is not None:
                    raise
                self._services.logger.warning(
                    f"Could not record the queue items {item_ids} of calibration run {run_id}; they run all the same, "
                    "and the status finds them by their origin",
                    exc_info=True,
                )
        return WorkingMemoryCalibrationRunCreated(run_id=run_id, device_class_id=device_class_id, item_ids=item_ids)

    def _prepare_run(
        self, model_key: str, request: WorkingMemoryCalibrationRequest, user_id: str, run_id: str
    ) -> "WorkingMemoryCalibrationRunCreated | _Enqueue":
        """The run a repeated request already enqueued, or what to enqueue."""
        key = request.idempotency_key
        found: Optional[tuple[str, CalibrationRun]] = None
        if key is not None and (found := self._find_run(run_id)) is not None:
            device_class_id, run = found
            # The run id does not derive from attempt_steps: a repeat that changes it is another request.
            if run.attempt_steps != request.attempt_steps:
                raise CalibrationAttemptConflictError(
                    f"Calibration run {run_id} plans {run.attempt_steps} runs for attempt {run.attempt_id}, and the "
                    f"repeat of its request {request.attempt_steps}"
                )
            if run.queue_item_ids:
                return WorkingMemoryCalibrationRunCreated(
                    run_id=run_id, device_class_id=device_class_id, item_ids=list(run.queue_item_ids)
                )
            # An earlier attempt may have queued the batch without recording its items. While the queue keeps the
            # key's receipt, enqueuing the batch again answers with the items it names, all of them. That needs
            # neither the model, which may have been removed since, nor the device. Without the receipt, enqueuing
            # again would queue the batch a second time: the items carrying the run's origin, and those the runner
            # saw, are taken over instead.
            if self._services.session_queue.get_enqueue_receipt(DEFAULT_QUEUE_ID, key, user_id) is not None:
                # The attempt that queued them marked the run, and the mark stays whatever this replay finds.
                return _Enqueue(self._batch(request, run_id), device_class_id, created=False, marks=False)
            item_ids = sorted(set(self._queued_item_ids(run_id)) | set(run.observed))
            if item_ids:
                self._record_items(device_class_id, run_id, item_ids)
                return WorkingMemoryCalibrationRunCreated(
                    run_id=run_id, device_class_id=device_class_id, item_ids=item_ids
                )
            if run.enqueue_attempted:
                # The batch may have been queued, and its items have left the queue unseen: enqueuing it again could
                # run it twice. The run stays without items and reads as canceled.
                self._services.logger.warning(
                    f"Calibration run {run_id} may have been enqueued, and the queue no longer has its items; a repeat "
                    "of its request enqueues nothing"
                )
                return WorkingMemoryCalibrationRunCreated(run_id=run_id, device_class_id=device_class_id, item_ids=[])
        elif key is not None:
            if self._was_reset(run_id):
                raise CalibrationRunNotFoundError(run_id, _RESET)
            # The key's receipt was written by an enqueue of a recorded run, and no run has this request's id: the
            # key belongs to another request. Refused here, before the model of this one is looked up.
            if self._services.session_queue.get_enqueue_receipt(DEFAULT_QUEUE_ID, key, user_id) is not None:
                raise EnqueueIdempotencyConflictError(f"Idempotency key {key} is already used by another submission")
        device = self._calibration_device()
        # Raises UnknownModelException for an unknown key.
        config = self._services.model_manager.store.get_model(model_key)
        if (changed := self._changed_models(request.graph)) is not None:
            raise CalibrationModelChangedError(changed)
        batch = self._batch(request, run_id)
        if found is not None:
            # The run is recorded without items and unmarked, and nothing of it is queued: enqueue its batch.
            return _Enqueue(batch, found[0], created=False, marks=True)
        device_class = self._device_class(device)
        env = self._environment(device)
        run = CalibrationRun(
            run_id=run_id,
            model_key=model_key,
            model_hash=getattr(config, "hash", None),
            user_id=user_id,
            created_at=_now(),
            env_id=env.id,
            attempt_id=request.attempt_id,
            attempt_steps=request.attempt_steps,
            soft=env.soft,
            model_identity=_model_identity(config),
            # Marked as it is recorded: the enqueue follows, and nothing else is written in between.
            enqueue_attempted=True,
        )

        def add(file: CalibrationStoreFile) -> tuple[bool, bool]:
            if run_id in file.removed_run_ids:
                raise CalibrationRunNotFoundError(run_id, _RESET)
            if any(known.run_id == run_id for known in file.runs):
                return False, False
            if (key := _attempt_key(run)) is not None:
                # The queue is read under the file's lock; no queue path takes it.
                if self._other_attempt_running(file, key):
                    raise CalibrationAttemptBusyError(
                        "Another calibration is still running on this GPU. Wait until it has finished, or stop it, "
                        "then calibrate again."
                    )
                # The attempt's earlier steps measured the model as it was then.
                for known in file.runs:
                    if _attempt_key(known) == key and known.model_key == model_key:
                        facts = _identity_changes(known, run.model_hash, _model_identity(config))
                        if facts:
                            raise CalibrationModelChangedError(
                                f"{getattr(config, 'name', model_key)} changed after this calibration was planned "
                                f"({', '.join(facts)}). Calibrate again."
                            )
                # Under the file's lock: two runs of one attempt that plan different counts cannot both be recorded.
                planned = {known.attempt_steps for known in file.runs if _attempt_key(known) == key}
                if (removed := _removed_steps(file).get(key)) is not None:
                    planned.add(removed.attempt_steps)
                if planned - {request.attempt_steps}:
                    raise CalibrationAttemptConflictError(
                        f"Attempt {request.attempt_id} plans {', '.join(sorted(map(str, planned)))} runs, and this "
                        f"run of it {request.attempt_steps}: every run of an attempt sends the same attempt_steps"
                    )
            file.runs.append(run)
            _add_env(file, env)
            return True, True

        created = bool(self._store.change(device_class.id, add, create=CalibrationDeviceClass.of(device_class)))
        self._run_files[run_id] = device_class.id
        return _Enqueue(batch, device_class.id, created=created, marks=created)

    def _other_attempt_running(self, file: CalibrationStoreFile, key: AttemptKey) -> bool:
        """Whether a run of another attempt than `key` (`_held_back`'s running: an item of it pending, in progress or
        waiting) is in the file. One attempt runs on a device class at a time: its runs measure the GPU back to back,
        each step on what the last one left, and another attempt's items in between would measure neither."""
        origins = self._services.session_queue.get_unfinished_origins(DEFAULT_QUEUE_ID, CALIBRATION_ORIGIN_PREFIX)
        running = {run_id for origin in origins if (run_id := calibration_run_id(origin)) is not None}
        return any(
            (other := _attempt_key(run)) is not None and other != key and run.run_id in running for run in file.runs
        )

    def _changed_models(self, graph: Graph) -> Optional[str]:
        """Why a model the graph names is not the one the graph was compiled against any more, or None: its record is
        gone, or names another hash, base or type than the graph does."""
        store = self._services.model_manager.store
        for named in _model_identifiers(graph):
            try:
                config = store.get_model(named.key)
            except UnknownModelException:
                return f"{named.name} was removed after this calibration was planned. Calibrate again."
            facts = [
                fact
                for fact, planned, current in (
                    ("hash", named.hash, getattr(config, "hash", None)),
                    ("base", str(named.base.value), str(getattr(config.base, "value", config.base))),
                    ("type", str(named.type.value), str(getattr(config.type, "value", config.type))),
                )
                if planned != current
            ]
            if facts:
                return f"{named.name} changed after this calibration was planned ({', '.join(facts)}). Calibrate again."
        return None

    @staticmethod
    def _batch(request: WorkingMemoryCalibrationRequest, run_id: str) -> Batch:
        try:
            return Batch(
                graph=request.graph,
                data=request.data,
                runs=1,
                origin=calibration_origin(run_id),
                destination=CALIBRATION_DESTINATION,
                idempotency_key=request.idempotency_key,
            )
        except ValueError as error:
            raise CalibrationGraphError(str(error)) from error

    def _was_reset(self, run_id: str) -> bool:
        for device_class_id in self._store.device_class_ids():
            read = self._store.read(device_class_id)
            if read is not None and read.file is not None and run_id in read.file.removed_run_ids:
                return True
        return False

    def _queued_item_ids(self, run_id: str) -> list[int]:
        """The queue items carrying the run's origin: its batch's, with their retries and workflow-call children."""
        queue = self._services.session_queue
        return sorted(queue.get_queue_item_ids(DEFAULT_QUEUE_ID, origin_prefix=calibration_origin(run_id)).item_ids)

    def _enqueued_nothing(self, device_class_id: str, run_id: str, created: bool, marks: bool) -> None:
        """The enqueue queued nothing: remove the run this request recorded, or clear the mark this request set on
        one it found, so that a repeat may enqueue the batch. A mark an earlier attempt set stays: that attempt may
        have queued the batch."""
        try:
            if created:
                self._remove_run(run_id, reset=False)
            elif marks:
                self._set_enqueue_attempted(device_class_id, run_id, False)
        except Exception:
            self._services.logger.warning(f"Could not discard calibration run {run_id}", exc_info=True)

    def _set_enqueue_attempted(self, device_class_id: str, run_id: str, attempted: bool) -> None:
        def mark(file: CalibrationStoreFile) -> tuple[bool, bool]:
            for run in file.runs:
                if run.run_id == run_id:
                    changed = run.enqueue_attempted != attempted
                    run.enqueue_attempted = attempted
                    return changed, True
            return False, False

        if not self._store.change(device_class_id, mark):
            raise CalibrationRunNotFoundError(run_id)

    def _record_items(self, device_class_id: str, run_id: str, item_ids: list[int]) -> None:
        def remember(file: CalibrationStoreFile) -> tuple[bool, bool]:
            for run in file.runs:
                if run.run_id == run_id:
                    changed = run.queue_item_ids != item_ids
                    run.queue_item_ids = list(item_ids)
                    return changed, True
            return False, False

        if not self._store.change(device_class_id, remember):
            # Reset while it was being created: its items fail at admission.
            raise CalibrationRunNotFoundError(run_id)

    def admit(self, queue_item: SessionQueueItem, run_id: str) -> Optional[str]:
        if self._services.configuration.multiuser:
            owner = self._services.users.get(queue_item.user_id)
            if owner is None or not owner.is_active or not owner.is_admin:
                return (
                    f"Calibration runs are for administrators only, and the owner of queue item {queue_item.item_id} "
                    "is not an active administrator."
                )
        try:
            device = self._calibration_device()
        except CalibrationUnavailableError as error:
            return str(error)
        found = self._find_run(run_id)
        if found is None:
            return f"Calibration run {run_id} does not exist; it was reset or never created."
        device_class_id, run = found
        if device_class_id != self._device_class(device).id:
            return (
                f"Calibration run {run_id} was created for device class {device_class_id}, and this machine's GPU "
                f"is of class {self._device_class(device).id}."
            )
        if (changed := self._model_changed_since(run, queue_item)) is not None:
            # The code reaches the run's summary: the observation the runner records next keeps it.
            self._observe(
                run_id,
                queue_item.item_id,
                ObservedItem(
                    status="failed",
                    error_message=changed,
                    error_code=MODEL_CHANGED,
                    retried_from_item_id=queue_item.retried_from_item_id,
                    observed_at=_now(),
                ),
            )
            return changed
        self._observe(
            run_id,
            queue_item.item_id,
            ObservedItem(
                status="in_progress", retried_from_item_id=queue_item.retried_from_item_id, observed_at=_now()
            ),
        )
        return None

    def _model_changed_since(self, run: CalibrationRun, queue_item: SessionQueueItem) -> Optional[str]:
        """Why the item would not measure the model its run was planned with, or None: the run's model is gone, or its
        hash, base, type, format or variant changed since the run was created; or a model the graph names changed
        since it was compiled (`_changed_models`). Runs stored before the rest was recorded compare the hash alone."""
        try:
            config = self._services.model_manager.store.get_model(run.model_key)
        except UnknownModelException:
            return f"Model {run.model_key} was removed after this calibration was planned. Calibrate again."
        name = getattr(config, "name", run.model_key)
        facts = _identity_changes(run, getattr(config, "hash", None), _model_identity(config))
        if facts:
            return f"{name} changed after this calibration was planned ({', '.join(facts)}). Calibrate again."
        session = getattr(queue_item, "session", None)
        return self._changed_models(session.graph) if session is not None else None

    def _observe(self, run_id: str, item_id: int, observation: ObservedItem) -> None:
        found = self._find_run(run_id)
        if found is None:
            return

        def record(file: CalibrationStoreFile) -> tuple[bool, None]:
            for run in file.runs:
                if run.run_id == run_id:
                    previous = run.observed.get(item_id)
                    # A later observation keeps what an earlier one knew and it does not: the item's lineage, and why
                    # admission failed it.
                    kept = {
                        field: getattr(previous, field)
                        for field in ("retried_from_item_id", "error_code")
                        if previous is not None and getattr(observation, field) is None
                    }
                    run.observed[item_id] = observation.model_copy(update=kept) if kept else observation
                    return True, None
            return False, None

        self._store.change(found[0], record)

    def observe(
        self,
        queue_item: SessionQueueItem,
        run_id: str,
        error: Optional[BaseException] = None,
        finished: Optional[SessionQueueItem] = None,
    ) -> None:
        try:
            current = self._services.session_queue.get_queue_item(queue_item.item_id)
        except SessionQueueItemNotFoundError:
            # A row that still exists has the last word. Without one: what the runner saw the queue settle (completed,
            # failed or canceled) before a prune or clear removed the item, which an error raised after that does not
            # undo; or the runner's own failure; or the item was removed before it finished (Clear Queue, a deleted
            # item).
            if finished is not None and finished.status not in _RUNNING:
                status, error_message = finished.status, finished.error_message
            elif error is not None:
                status, error_message = "failed", f"{type(error).__name__}: {error}"
            else:
                status, error_message = "canceled", "The queue item was removed from the queue."
            self._observe(
                run_id,
                queue_item.item_id,
                ObservedItem(
                    status=status,
                    error_message=error_message,
                    retried_from_item_id=queue_item.retried_from_item_id,
                    observed_at=_now(),
                ),
            )
            return
        status, error_message = current.status, current.error_message
        if error is not None and status in _RUNNING:
            # The processor fails the item after the runner raised; recorded here, a later prune cannot lose it.
            status, error_message = "failed", f"{type(error).__name__}: {error}"
        self._observe(
            run_id,
            current.item_id,
            ObservedItem(
                status=status,
                error_message=error_message,
                retried_from_item_id=current.retried_from_item_id,
                observed_at=_now(),
            ),
        )

    async def _on_queue_items_retried(self, event: FastAPIEvent[QueueItemsRetriedEvent]) -> None:
        """Record the calibration retries the queue announces on their runs (`record_retries`), off the event loop."""
        retried = list(event[1].retried_items)
        if not self._listening or not retried:
            return
        try:
            await asyncio.to_thread(self.record_retries, retried)
        except Exception:
            self._services.logger.warning("Could not record the retries of calibration items", exc_info=True)

    def record_retries(self, retried: Iterable[RetriedQueueItem]) -> None:
        """Record each retry of a calibration item on its run, as an item pending that stands for the retried item's
        lineage, under the store file's lock. So the store knows every item of every run: a retry canceled and pruned
        before anything else saw it still reads as canceled (`_canceled_runs`), and its attempt as aborted.

        A retry belongs to the run whose origin it carries: only the server creates items with a calibration origin, and
        a retry inherits its source's, so the origin alone names the run. Not the source: a retry of a retry comes from
        the earlier retry, whose announcement may be handled after its own. A retry the runner already saw keeps its
        observation. A retry announced while this process was down, or whose event was lost to a crash between the
        retry's commit and this handler, is recorded once the runner admits it (`admit`); one canceled before that
        leaves its run on the outcome of the item it retried.

        One read of the retries' origins; only a calibration retry is read whole, for the item it retried. A retry that
        cannot be recorded is logged and the others still are."""
        queue = self._services.session_queue
        retried_ids = [retry.item_id for retry in retried]
        for summary in queue.get_queue_item_summaries_by_ids(DEFAULT_QUEUE_ID, retried_ids):
            run_id = calibration_run_id(summary.origin)
            if run_id is None:
                continue
            try:
                found = self._find_run(run_id)
                if found is None:
                    continue
                item = queue.get_queue_item(summary.item_id)
                self._store.change(found[0], partial(_record_retry, run_id, item))
            except SessionQueueItemNotFoundError:
                continue
            except Exception:
                self._services.logger.warning(
                    f"Could not record the retry {summary.item_id} of calibration run {run_id}", exc_info=True
                )

    def model_reset_epochs(self, run_id: str) -> dict[str, int]:
        found = self._find_run(run_id)
        if found is None:
            return {}
        read = self._store.read(found[0])
        return dict(read.file.model_reset_epochs) if read is not None and read.file is not None else {}

    def record_windows(
        self, run_id: str, windows: list[MeasuredWindow], model_reset_epochs: Optional[Mapping[str, int]] = None
    ) -> None:
        if model_reset_epochs is None:
            model_reset_epochs = self.model_reset_epochs(run_id)
        found = self._find_run(run_id)
        if found is None:
            self._services.logger.warning(
                f"Dropped {len(windows)} working-memory measurements of calibration run {run_id}: the run was reset."
            )
            return
        run_class_id = found[0]
        by_device: dict[str, list[MeasuredWindow]] = {}
        for window in windows:
            by_device.setdefault(window.device, []).append(window)
        for device_name, group in by_device.items():
            device = torch.device(device_name)
            device_class = self._device_class(device)
            if device_class.id != run_class_id:
                # Admission made the run's device class the generation device's; a window elsewhere is a model the
                # node ran on another device, which this run does not calibrate.
                self._services.logger.warning(
                    f"Dropped {len(group)} working-memory measurements of calibration run {run_id}: they ran on "
                    f"{device_name} (device class {device_class.id}), not on the run's device class {run_class_id}."
                )
                continue
            env = self._environment(device)
            points = [CalibrationPoint(**window.model_dump(), env_id=env.id) for window in group]
            add = partial(_add_points, run_id=run_id, env=env, points=points, epochs=model_reset_epochs)
            dropped = len(points) - (self._store.change(run_class_id, add) or 0)
            if dropped:
                self._services.logger.warning(
                    f"Dropped {dropped} working-memory measurements of calibration run {run_id}: the run, or the "
                    "model they measured, was reset."
                )

    # endregion

    # region Status

    def _live_items(self) -> dict[str, dict[int, QueueItemLineage]]:
        """The queue's calibration items, per run, as the queue has them now."""
        live: dict[str, dict[int, QueueItemLineage]] = {}
        for item in self._services.session_queue.get_queue_item_lineage(DEFAULT_QUEUE_ID, CALIBRATION_ORIGIN_PREFIX):
            # The queue's origin filter ignores case; only the exact prefix is the server's.
            run_id = calibration_run_id(item.origin)
            if run_id is not None:
                live.setdefault(run_id, {})[item.item_id] = item
        return live

    def get_status(self) -> WorkingMemoryCalibrationStatus:
        current_class: Optional[DeviceClass] = None
        current_env: Optional[Environment] = None
        unavailable: Optional[str] = None
        try:
            device = self._calibration_device()
            current_class = self._device_class(device)
            current_env = self._environment(device)
        except CalibrationUnavailableError as error:
            unavailable = str(error)

        live = self._live_items()
        present = self._present_backends()
        device_class_ids = set(self._store.device_class_ids())
        if current_class is not None:
            device_class_ids.add(current_class.id)
        device_classes = []
        for device_class_id in sorted(device_class_ids):
            is_current = current_class is not None and device_class_id == current_class.id
            read = self._store.read(device_class_id)
            if read is None:
                if current_class is None or not is_current:
                    continue
                empty = CalibrationStoreFile(
                    schema_version=STORE_SCHEMA_VERSION, device_class=CalibrationDeviceClass.of(current_class)
                )
                read = StoreRead(device_class_id, empty, None)
            if is_current:
                # Creating a run writes this file: an unusable one is kept as it is, and the root may refuse writes.
                problem = read.problem if read.file is None else self._store.write_problem(device_class_id)
                if problem is not None:
                    unavailable = f"The calibration store of this GPU's device class cannot be written: {problem}"
            device_classes.append(
                self._device_class_status(
                    read, is_current, current_env if is_current else None, live, present.get(device_class_id)
                )
            )
        return WorkingMemoryCalibrationStatus(
            available=unavailable is None,
            unavailable_reason=unavailable,
            current_device_class_id=current_class.id if current_class is not None else None,
            current_env_id=current_env.id if current_env is not None else None,
            device_classes=device_classes,
        )

    @staticmethod
    def _device_class_status(
        read: StoreRead,
        is_current: bool,
        current_env: Optional[Environment],
        live: Mapping[str, Mapping[int, QueueItemLineage]],
        backend: Optional[Backend],
    ) -> WorkingMemoryCalibrationDeviceClass:
        method, verified = _measurement(backend)
        file = read.file
        if file is None:
            return WorkingMemoryCalibrationDeviceClass(
                device_class_id=read.device_class_id,
                device_class=None,
                usable=False,
                problem=read.problem,
                is_current=is_current,
                measurement_method=method,
                measurement_verified=verified,
                runs=[],
                point_counts=[],
                envs=[],
            )
        points_per_run: dict[str, int] = {}
        valid_points_per_run: dict[str, int] = {}
        points_per_env: dict[str, int] = {}
        counts: dict[tuple[str, int, bool, bool], int] = {}
        for point in file.points:
            points_per_run[point.run_id] = points_per_run.get(point.run_id, 0) + 1
            if point.valid:
                valid_points_per_run[point.run_id] = valid_points_per_run.get(point.run_id, 0) + 1
            points_per_env[point.env_id] = points_per_env.get(point.env_id, 0) + 1
            key = (point.estimator, point.version, point.valid, point.fit_eligible)
            counts[key] = counts.get(key, 0) + 1
        runs = []
        for run in file.runs:
            point_count = points_per_run.get(run.run_id, 0)
            status, error, error_code, item_ids = derive_run_status(
                run, live.get(run.run_id, {}), valid_points_per_run.get(run.run_id, 0)
            )
            runs.append(
                WorkingMemoryCalibrationRunSummary(
                    run_id=run.run_id,
                    model_key=run.model_key,
                    model_hash=run.model_hash,
                    user_id=run.user_id,
                    created_at=run.created_at,
                    env_id=run.env_id,
                    status=status,
                    error=error,
                    error_code=error_code,
                    item_ids=item_ids,
                    point_count=point_count,
                    attempt_id=run.attempt_id,
                    attempt_steps=run.attempt_steps,
                )
            )
        envs = [
            WorkingMemoryCalibrationEnvironment(
                id=env.id,
                hard=env.hard,
                soft=env.soft,
                hard_matches_current=None if current_env is None else env.id == current_env.id,
                soft_matches_current=None if current_env is None else env.soft == current_env.soft,
                point_count=points_per_env.get(env.id, 0),
            )
            for env in file.envs
        ]
        return WorkingMemoryCalibrationDeviceClass(
            device_class_id=read.device_class_id,
            device_class=file.device_class,
            usable=True,
            problem=None,
            is_current=is_current,
            measurement_method=method,
            measurement_verified=verified,
            runs=runs,
            point_counts=[
                WorkingMemoryCalibrationPointCount(
                    estimator=estimator, version=version, valid=valid, fit_eligible=fit_eligible, count=count
                )
                for (estimator, version, valid, fit_eligible), count in sorted(counts.items())
            ],
            envs=envs,
        )

    # endregion

    # region User fits

    def _user_fit_for(self, spec: EstimatorSpec, platform: Platform, device: torch.device) -> Optional[UserFit]:
        """What estimates ask (`user_fit.user_fit_for`): a lookup in memory, no file or device access."""
        found = self._fits.get(_device_key(device))
        if found is None or found.platform != platform:
            return None
        return found.fit

    @staticmethod
    def _signature(path: str) -> FileSignature:
        try:
            stat = os.stat(path)
        except FileNotFoundError:
            return None
        return stat.st_mtime_ns, stat.st_size, stat.st_ino

    def refresh_user_fits(self) -> None:
        with self._fits_lock:
            fits: dict[str, _DeviceFit] = {}
            for device in self._devices():
                if backend_for(device) is None:
                    continue
                device_class = self._device_class(device)
                env = self._environment(device)
                platform = platform_for(device)
                signature = self._signature(str(self._store.path_for(device_class.id)))
                key = _device_key(device)
                known = self._fits.get(key)
                file: Optional[CalibrationStoreFile] = None
                held_back: Optional[HeldBack] = None
                if known is not None and (known.signature, known.env_id, known.platform) == (
                    signature,
                    env.id,
                    platform,
                ):
                    if known.file is None:
                        fits[key] = known
                        continue
                    # The same file: only the queue (an item that finished, a retry queued) can change the runs it
                    # leaves out.
                    held_back = self._held_back(known.file)
                    if held_back == known.held_back:
                        fits[key] = known
                        continue
                    file = known.file
                elif signature is not None:
                    read = self._store.read(device_class.id)
                    if read is not None and read.file is None:
                        self._services.logger.warning(
                            f"Working-memory estimates on {device} keep the shipped values: {read.problem}"
                        )
                    file = read.file if read is not None else None
                if held_back is None:
                    held_back = self._held_back(file) if file is not None else {}
                fit = user_fit_of(file, env.id, platform, held_back=held_back)[0] if file is not None else None
                fits[key] = _DeviceFit(
                    signature=signature, env_id=env.id, platform=platform, fit=fit, file=file, held_back=held_back
                )
            self._fits = fits

    def _held_back(self, file: CalibrationStoreFile) -> dict[str, CalibrationHoldBack]:
        """The runs whose windows no fit takes, and why, judged per attempt (an owner's `attempt_id`, `AttemptKey`; a
        run without one is an attempt of its own):

        - running: the queue holds an item of one of the attempt's runs that is pending, in progress or waiting. An
          item carries its run's origin from the moment it is queued, before the run records it or the runner admits
          it, and so does a retry: a run that failed holds its windows back again while its retry waits, and stops once
          the retry finished or was canceled or removed. A client posts each step of an attempt as its own run: the
          finished earlier steps must not price the later, larger ones either.
        - aborted_attempt: none of its runs is running, and the attempt (one owner's `attempt_id`) was aborted. Either
          an item of it ended canceled (`_canceled_runs`: Stop cancels what the queue still holds, and a queue clear
          removes it); or it queued fewer runs than it planned (`_queued_runs`): Stop, or a submission that failed,
          ended it before it queued them all; or a reset removed a run of it while it ran or after it was aborted
          (`RemovedAttemptSteps.aborted`). Either way it measured only some of its sizes, and it never feeds a fit. A
          step not posted yet looks the same until it is. An item that failed (an out-of-memory error at the largest
          size) ends that step, not the attempt. A run without an attempt is never aborted: canceled, it feeds what it
          measured, as before.

        A run a reset removed counts as queued (`removed_attempt_steps`), so removing a run of a finished attempt
        discards only its own points.

        One read of the queue per call (`get_unfinished_origins`), which costs the queue's unfinished items, not its
        history; and the reads by id `_canceled_runs` makes. None for a file without runs."""
        if not file.runs:
            return {}
        origins = self._services.session_queue.get_unfinished_origins(DEFAULT_QUEUE_ID, CALIBRATION_ORIGIN_PREFIX)
        # The queue's origin filter ignores case; only the exact prefix is the server's.
        running = {run_id for origin in origins if (run_id := calibration_run_id(origin)) is not None}
        attempts: dict[tuple[Optional[AttemptKey], str], list[CalibrationRun]] = {}
        for run in file.runs:
            key = _attempt_key(run)
            attempts.setdefault((key, "" if key is not None else run.run_id), []).append(run)
        notes = _removed_steps(file)
        held_back: dict[str, CalibrationHoldBack] = {}

        def hold(runs: list[CalibrationRun], reason: CalibrationHoldBack) -> None:
            held_back.update(dict.fromkeys((run.run_id for run in runs), reason))

        settled: list[list[CalibrationRun]] = []
        for (key, _), runs in attempts.items():
            is_attempt = key is not None
            removed = notes.get(key) if key is not None else None
            queued, planned = _queued_runs(runs, removed)
            if any(run.run_id in running for run in runs):
                hold(runs, "running")
            elif (removed is not None and removed.aborted) or queued < planned:
                hold(runs, "aborted_attempt")
            elif is_attempt:
                # A run without an attempt is judged on its own, as before: a canceled one feeds what it measured.
                settled.append(runs)
        canceled, started = self._canceled_runs([run for runs in settled for run in runs])
        for runs in settled:
            if any(run.run_id in started for run in runs):
                # An item of it the store records runs after all: a retry queued since the read for running items.
                hold(runs, "running")
            elif any(run.run_id in canceled for run in runs):
                hold(runs, "aborted_attempt")
        return held_back

    def _canceled_runs(self, runs: list[CalibrationRun]) -> tuple[set[str], set[str]]:
        """The runs among `runs`, none of them running when `_held_back` asked the queue, whose attempt an item of
        theirs interrupted: the item that stands for one of its lineages, its latest retry or itself
        (`_lineage_outcomes`), was canceled, or is gone from the queue before the runner saw it finish, or admission
        failed it because a model changed after the calibration was planned (`error_code` `model_changed`: the rest of
        the attempt would measure another model); and a run with no item at all, whose items were never recorded.
        Judged per item, so such an item counts beside a failed one, which decides a run's displayed status; an item
        that failed otherwise (out of memory at the largest size) ends that step, not the attempt; and an item whose
        retry completed or failed no longer counts. And the runs one of whose items the queue shows running after all:
        they are not judged yet.

        The items are those the store records: the run's batch, and every retry (`record_retries`, and the runner's
        observations). The runner's observation decides an item it saw finish. Any other is read from the queue by id
        once it finished, kept for the process's life since a finished item never changes, and only as the queue holds
        it with the run's origin: a recreated or restored database gives the id to an unrelated item, and the run's own
        is gone. Reads the queue only for items without such an outcome."""
        if not runs:
            return set(), set()
        unknown: dict[int, CalibrationRun] = {
            item_id: run
            for run in runs
            for item_id in set(run.queue_item_ids) | set(run.observed)
            if item_id not in self._item_outcomes
            and ((observed := run.observed.get(item_id)) is None or observed.status in _RUNNING)
        }
        started: set[str] = set()
        if unknown:
            queue = self._services.session_queue
            found = {
                summary.item_id: summary
                for summary in queue.get_queue_item_summaries_by_ids(DEFAULT_QUEUE_ID, list(unknown))
            }
            for item_id, run in unknown.items():
                summary = found.get(item_id)
                if summary is None or summary.origin != calibration_origin(run.run_id):
                    self._item_outcomes[item_id] = "canceled"
                elif summary.status in _RUNNING:
                    # Defensive: every item read here is the store's, of a run the read for unfinished items found
                    # settled, and a settled item does not start again (a retry is a new item, which that read would
                    # have found). Should one run all the same, its status is not final: it must not enter the
                    # process-lifetime outcomes, and the run waits for the next refresh.
                    started.add(run.run_id)
                else:
                    self._item_outcomes[item_id] = summary.status
        canceled: set[str] = set()
        for run in runs:
            if run.run_id in started:
                continue
            outcomes = {
                item_id: QueueItemLineage(
                    item_id=item_id,
                    status=self._item_outcomes[item_id],
                    origin=None,
                    retried_from_item_id=observed.retried_from_item_id
                    if (observed := run.observed.get(item_id)) is not None
                    else None,
                )
                for item_id in set(run.queue_item_ids) | set(run.observed)
                if item_id in self._item_outcomes
            }
            _, statuses = _lineage_outcomes(run, outcomes)
            if not statuses or any(_interrupts(run, item_id, status) for item_id, status in statuses.items()):
                canceled.add(run.run_id)
        return canceled, started

    def _try_refresh_user_fits(self, when: str) -> None:
        """Refresh the fits `when` (for the log). A store that cannot be read now must not fail what triggered it, a
        reset that already happened or a plan; the fits read so far then stay until the next refresh."""
        try:
            self.refresh_user_fits()
        except Exception:
            self._services.logger.warning(f"Could not refresh the working-memory user fits {when}", exc_info=True)

    # endregion

    # region Plan and summary

    def get_plan(self, model_key: str) -> WorkingMemoryCalibrationPlan:
        store = self._services.model_manager.store
        # Raises UnknownModelException for an unknown key.
        target = store.get_model(model_key)
        # The case sizes price the decode with the fits estimates read: the store as it is now, not as the last
        # queue item found it.
        self._try_refresh_user_fits("before building a calibration plan")
        device: Optional[torch.device] = None
        unavailable: Optional[str] = None
        try:
            device = self._calibration_device()
        except CalibrationUnavailableError as error:
            unavailable = str(error)

        def reservation(estimator: str, height: int, width: int, dtype: str) -> int:
            assert device is not None
            return self._untiled_decode_reservation(estimator, height, width, dtype, device)

        return build_plan(
            target,
            store.all_models(),
            device_class=self._device_class(device) if device is not None else None,
            unavailable=unavailable,
            platform=platform_for(device) if device is not None else None,
            precision=self._services.configuration.precision,
            vae_dtypes=self._vae_dtypes_on(device),
            reservation=reservation,
            specs=import_estimator_modules(),
            capacity_bytes=self._decode_capacity(device) if device is not None else None,
            working_memory_floor_bytes=int(self._services.configuration.device_working_mem_gb * 2**30),
            force_tiled_decode=self._services.configuration.force_tiled_decode,
        )

    def _vae_dtypes_on(self, device: Optional[torch.device]) -> VaeDtypes:
        """What the VAE loaders choose on `device`, by default the one models load onto."""
        return self._vae_dtypes(device or TorchDevice.choose_torch_device(), self._services.configuration.models_path)

    def _cache_memory(self, device: torch.device) -> DeviceMemory:
        """`DeviceMemory` for a GPU `device`: the capacity its model cache gives a calibration's lock, which that lock
        then asks the same cache for."""
        load = self._services.model_manager.load
        cache = load.ram_caches.get(str(TorchDevice.normalize(device))) or load.ram_cache
        reading = read_counters(device)
        outside = None if reading.driver is None or reading.reserved is None else reading.driver - reading.reserved
        return DeviceMemory(capacity_bytes=cache.calibration_capacity(), outside_torch_bytes=outside)

    def _decode_capacity(self, device: torch.device) -> int:
        """What the model cache can give one decode and its VAE on `device` (`DeviceMemory.capacity_bytes`): what a
        calibration's lock gets once its pre-lock offload ran, the cache's reserves and cap applied.

        This process's memory outside torch's allocator counts at the larger of what it is now and the most a stored
        window of this environment found it at (`recorded_outside_torch_bytes`): it grows with the first generations
        (on ROCm under Windows, HIP's retained pool and MIOpen's workspaces, from about 0.3 GB in a fresh server to
        2.6 GB on an RX 9060 XT), so a fresh server's reading alone overstates what a decode will find. Without a stored
        window, a first calibration sizes for the reading alone. Where the driver gives no figure now, the recorded one
        counts whole. The store is the one `refresh_user_fits` read last; nothing here reads it again."""
        memory = self._device_memory(device)
        capacity = memory.capacity_bytes
        known = self._fits.get(_device_key(device))
        if known is not None and known.file is not None:
            recorded = recorded_outside_torch_bytes(known.file, known.env_id)
            if recorded is not None:
                capacity -= max(0, recorded - (memory.outside_torch_bytes or 0))
        return capacity

    def _present_backends(self) -> dict[str, Backend]:
        """The backend of each device class a GPU generation device here belongs to, by device class id."""
        present: dict[str, Backend] = {}
        for device in self._devices():
            backend = backend_for(device)
            if backend is not None:
                present[self._device_class(device).id] = backend
        return present

    def _present_device(self, device_class_id: str) -> Optional[torch.device]:
        """A GPU generation device of this class; None when no generation device is one."""
        for device in self._devices():
            if backend_for(device) is not None and self._device_class(device).id == device_class_id:
                return device
        return None

    def _present(self, device_class_id: str) -> Optional[tuple[Environment, Platform]]:
        """The environment and platform of a GPU generation device of this class; None when no such device is one."""
        device = self._present_device(device_class_id)
        return None if device is None else (self._environment(device), platform_for(device))

    def get_model_summary(
        self, model_key: str, device_class_id: Optional[str] = None
    ) -> WorkingMemoryCalibrationModelSummary:
        store = self._services.model_manager.store
        # Raises UnknownModelException for an unknown key.
        target = store.get_model(model_key)
        class_id = self._current_device_class_id(device_class_id)
        read = self._store.read(class_id)
        present = self._present(class_id)
        method, verified = _measurement(self._present_backends().get(class_id))
        summary = WorkingMemoryCalibrationModelSummary(
            model_key=model_key,
            device_class_id=class_id,
            usable=read is None or read.file is not None,
            problem=read.problem if read is not None else None,
            current_env_id=present[0].id if present is not None else None,
            measurement_method=method,
            measurement_verified=verified,
            lines=[],
            excluded_windows=0,
            runs=[],
        )
        if read is None or read.file is None:
            return summary
        file = read.file
        runs = [run for run in file.runs if run.model_key == model_key]
        run_ids = {run.run_id for run in runs}

        def name_of(key: str) -> Optional[str]:
            try:
                return getattr(store.get_model(key), "name", None)
            except UnknownModelException:
                return None

        def platform_of(env: CalibrationEnvironment) -> Optional[Platform]:
            if present is not None and env.id == present[0].id:
                return present[1]
            try:
                return platform_for_environment(file.device_class.backend, env.hard)
            except ValueError:
                return None

        held_back = self._held_back(file)
        lines, excluded = model_lines(
            file,
            model_key,
            present[0] if present is not None else None,
            platform_of,
            import_estimator_modules(),
            priced_on=decode_lines_of(
                target,
                self._services.configuration.precision,
                self._vae_dtypes_on(self._present_device(class_id)),
                store.all_models(),
            ),
            name_of=name_of,
            held_back=held_back,
        )
        points_per_run: dict[str, int] = {}
        valid_points_per_run: dict[str, int] = {}
        for point in file.points:
            if point.run_id in run_ids:
                points_per_run[point.run_id] = points_per_run.get(point.run_id, 0) + 1
                if point.valid:
                    valid_points_per_run[point.run_id] = valid_points_per_run.get(point.run_id, 0) + 1
        live = self._live_items() if runs else {}
        model_runs = []
        for run in sorted(runs, key=lambda run: run.created_at, reverse=True):
            point_count = points_per_run.get(run.run_id, 0)
            status, error, error_code, _ = derive_run_status(
                run, live.get(run.run_id, {}), valid_points_per_run.get(run.run_id, 0)
            )
            model_runs.append(
                WorkingMemoryCalibrationModelRun(
                    run_id=run.run_id,
                    status=status,
                    error=error,
                    error_code=error_code,
                    created_at=run.created_at,
                    point_count=point_count,
                    attempt_id=run.attempt_id,
                    attempt_steps=run.attempt_steps,
                    held_back=held_back.get(run.run_id),
                )
            )
        return summary.model_copy(update={"lines": lines, "excluded_windows": excluded, "runs": model_runs})

    # endregion

    # region Reset and export

    @staticmethod
    def _remove(
        file: CalibrationStoreFile,
        run_ids: set[str],
        model_key: Optional[str] = None,
        *,
        reasons: Optional[Mapping[str, Optional[CalibrationHoldBack]]],
    ) -> tuple[bool, WorkingMemoryCalibrationResetResult]:
        """Remove the runs `run_ids` with their points, and with `model_key` every point the summary attributes to
        that model (`measured_models`). With `reasons` (`_hold_reasons` of this `file`), for a reset: keep the removed
        runs' ids, so that a repeat of the request that started one cannot start it again, and note what the reset
        removed of each attempt (`_note_removed_steps`). A creation that queued nothing passes None and removes its run
        without either."""
        models = run_models(file)
        removed = [run for run in file.runs if run.run_id in run_ids]
        if reasons is not None:
            file.removed_run_ids.extend(run.run_id for run in removed if _is_keyed(run.run_id))
        runs = [run for run in file.runs if run.run_id not in run_ids]
        points = [
            point
            for point in file.points
            if point.run_id not in run_ids and (model_key is None or model_key not in measured_models(point, models))
        ]
        result = WorkingMemoryCalibrationResetResult(
            runs_removed=len(file.runs) - len(runs), points_removed=len(file.points) - len(points)
        )
        file.runs, file.points = runs, points
        if reasons is not None:
            _note_removed_steps(file, removed, reasons)
        _prune_envs(file)
        return result.runs_removed > 0 or result.points_removed > 0, result

    def _hold_reasons(self, file: CalibrationStoreFile) -> dict[str, Optional[CalibrationHoldBack]]:
        """Per stored run of `file`, why its windows are held back now (`_held_back`; None where they are not). Read by
        a reset inside its change of the file, under the file's lock, which the retries the queue announces take too
        (`record_retries`): none slips in between. The queue is read under that lock; no queue path takes it."""
        held_back = self._held_back(file)
        return {run.run_id: held_back.get(run.run_id) for run in file.runs}

    def reset_run(self, run_id: str) -> WorkingMemoryCalibrationResetResult:
        return self._remove_run(run_id, reset=True)

    def _remove_run(self, run_id: str, *, reset: bool) -> WorkingMemoryCalibrationResetResult:
        found = self._find_run(run_id)
        if found is None:
            raise CalibrationRunNotFoundError(run_id)

        def remove(file: CalibrationStoreFile) -> tuple[bool, WorkingMemoryCalibrationResetResult]:
            return self._remove(file, {run_id}, reasons=self._hold_reasons(file) if reset else None)

        result = self._store.change(found[0], remove)
        self._run_files.pop(run_id, None)
        self._try_refresh_user_fits("after a reset")
        return result or WorkingMemoryCalibrationResetResult(runs_removed=0, points_removed=0)

    def reset_model(self, model_key: str, device_class_id: Optional[str] = None) -> WorkingMemoryCalibrationResetResult:
        class_id = self._current_device_class_id(device_class_id)

        def remove(file: CalibrationStoreFile) -> tuple[bool, WorkingMemoryCalibrationResetResult]:
            reasons = self._hold_reasons(file)
            # Written with the removal or not at all: a node that read the older epoch drops its points of the model.
            file.model_reset_epochs[model_key] = file.model_reset_epochs.get(model_key, 0) + 1
            run_ids = {run.run_id for run in file.runs if run.model_key == model_key}
            return True, self._remove(file, run_ids, model_key, reasons=reasons)[1]

        result = self._store.change(class_id, remove)
        self._try_refresh_user_fits("after a reset")
        return result or WorkingMemoryCalibrationResetResult(runs_removed=0, points_removed=0)

    def reset_all(self, device_class_id: Optional[str] = None) -> WorkingMemoryCalibrationResetResult:
        class_id = self._current_device_class_id(device_class_id)

        def remove(file: CalibrationStoreFile) -> tuple[bool, WorkingMemoryCalibrationResetResult]:
            result = WorkingMemoryCalibrationResetResult(runs_removed=len(file.runs), points_removed=len(file.points))
            changed = bool(file.runs or file.points or file.envs)
            self._remove(file, {run.run_id for run in file.runs}, reasons=self._hold_reasons(file))
            file.points, file.envs = [], []
            return changed, result

        result = self._store.change(class_id, remove)
        self._try_refresh_user_fits("after a reset")
        return result or WorkingMemoryCalibrationResetResult(runs_removed=0, points_removed=0)

    def export(self, device_class_id: Optional[str] = None) -> Optional[bytes]:
        return self._store.read_raw(self._current_device_class_id(device_class_id))

    # endregion
