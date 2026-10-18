import asyncio
import datetime
import os
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import partial
from itertools import chain
from typing import TYPE_CHECKING, Final, Optional

import torch

from invokeai.app.invocations.call_saved_workflow import CallSavedWorkflowInvocation
from invokeai.app.services.model_records import UnknownModelException
from invokeai.app.services.session_queue.session_queue_common import (
    DEFAULT_QUEUE_ID,
    QUEUE_ITEM_STATUS,
    Batch,
    SessionQueueItem,
    SessionQueueItemNotFoundError,
)
from invokeai.app.services.working_memory_calibration.working_memory_calibration_base import (
    WorkingMemoryCalibrationServiceBase,
)
from invokeai.app.services.working_memory_calibration.working_memory_calibration_common import (
    CALIBRATION_DESTINATION,
    CALIBRATION_ORIGIN_PREFIX,
    STORE_SCHEMA_VERSION,
    CalibrationDeviceClass,
    CalibrationEnvironment,
    CalibrationGraphError,
    CalibrationPoint,
    CalibrationQueueFullError,
    CalibrationRun,
    CalibrationRunNotFoundError,
    CalibrationRunStatus,
    CalibrationStoreFile,
    CalibrationUnavailableError,
    ObservedItem,
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
    measured_models,
    model_lines,
    run_models,
    user_fit_of,
)
from invokeai.app.services.working_memory_calibration.working_memory_calibration_plan import (
    build_plan,
    decode_lines_of,
)
from invokeai.app.services.working_memory_calibration.working_memory_calibration_store import (
    CalibrationStore,
    StoreRead,
)
from invokeai.app.util.misc import uuid_string
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.vae_working_memory import estimate_untiled_vae_decode
from invokeai.backend.working_memory import fingerprint, user_fit
from invokeai.backend.working_memory.counters import Backend, backend_for, driver_method, method_verified
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

_SUMMARY_CHUNK: Final = 500
"""Queue items read per summary query when the status collects the calibration items."""

_RUN_ID_NAMESPACE: Final = uuid.UUID("3c1f6a52-8d0e-4b6f-9a57-2f0d5c8e41b3")
"""Derives a run id from an owner and a retry key."""

_RUNNING: Final[frozenset[QUEUE_ITEM_STATUS]] = frozenset({"pending", "in_progress", "waiting"})

SettingValue = str | int | float | bool | None


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _untiled_decode_reservation(estimator: str, height: int, width: int, dtype: str, device: torch.device) -> int:
    return estimate_untiled_vae_decode(estimator, height, width, dtype, device).reservation_bytes


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
    """The file the fit was made from, kept to re-fit it when only the runs still running changed."""
    live_runs: frozenset[str]
    """The runs whose windows the fit left out, because they were still running."""
    settled_items: frozenset[int]
    """Calibration queue items found finished in the queue, which `_live_runs` asks about no more."""


def _add_env(file: CalibrationStoreFile, env: Environment) -> None:
    if all(known.id != env.id for known in file.envs):
        file.envs.append(CalibrationEnvironment.of(env))


def _prune_envs(file: CalibrationStoreFile) -> None:
    used = {run.env_id for run in file.runs} | {point.env_id for point in file.points}
    file.envs = [env for env in file.envs if env.id in used]


def _add_points(
    file: CalibrationStoreFile, *, run_id: str, env: Environment, points: list[CalibrationPoint]
) -> tuple[bool, bool]:
    """Attach `points` to their run; nothing when the file does not hold the run."""
    if all(run.run_id != run_id for run in file.runs):
        return False, False
    _add_env(file, env)
    file.points.extend(points)
    return True, True


def derive_run_status(
    run: CalibrationRun, live: Mapping[int, QUEUE_ITEM_STATUS], point_count: int
) -> tuple[CalibrationRunStatus, Optional[str], list[int]]:
    """A run's status, error and items, from its items' statuses: `live` as the queue has them now, the run's own
    observations for items the queue no longer has. An item the queue no longer has counts as canceled unless the
    runner saw it finish. A retry stands in for the item it retried."""
    item_ids = sorted(set(run.queue_item_ids) | set(run.observed) | set(live))

    def status_of(item_id: int) -> QUEUE_ITEM_STATUS:
        if item_id in live:
            return live[item_id]
        observed = run.observed.get(item_id)
        # Gone from the queue while the runner last saw it running (a crash, a cleared queue): it will not finish.
        if observed is None or observed.status in _RUNNING:
            return "canceled"
        return observed.status

    latest: dict[int, int] = {}
    for item_id in item_ids:
        observed = run.observed.get(item_id)
        lineage = observed.retried_from_item_id if observed and observed.retried_from_item_id else item_id
        latest[lineage] = max(latest.get(lineage, item_id), item_id)
    effective = sorted(latest.values())
    statuses = {item_id: status_of(item_id) for item_id in effective}

    if not effective:
        return "canceled", None, item_ids
    if any(status in _RUNNING for status in statuses.values()):
        return "running", None, item_ids
    failed = [item_id for item_id, status in statuses.items() if status == "failed"]
    if failed:
        observed = run.observed.get(failed[0])
        error = observed.error_message if observed is not None and observed.error_message else None
        return "failed", error or f"Queue item {failed[0]} failed", item_ids
    if all(status == "completed" for status in statuses.values()):
        return "completed", None, item_ids
    return ("partial" if point_count > 0 else "canceled"), None, item_ids


class WorkingMemoryCalibrationService(WorkingMemoryCalibrationServiceBase):
    """Calibration runs and their store.

    `generation_devices`, `device_class_for` and `environment_for` default to the machine's own; tests pass their own
    because a device class can only be read from real hardware.
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
    ) -> None:
        self._store = store
        self._generation_devices = generation_devices
        self._device_class_for = device_class_for
        self._environment_for = environment_for
        self._untiled_decode_reservation = untiled_decode_reservation
        # A card's class cannot change while the process runs; reading it touches the driver.
        self._device_classes: dict[str, DeviceClass] = {}
        # Which file holds a run, so admitting an item and recording its windows do not scan every file.
        self._run_files: dict[str, str] = {}
        self._invoker: Optional["Invoker"] = None
        # The user fit per generation device: replaced whole by `refresh_user_fits`, read by every estimate.
        self._fits: Mapping[str, _DeviceFit] = {}
        self._fits_lock = threading.Lock()
        # One object, so that `stop` unregisters exactly what `start` registered.
        self._provider: user_fit.UserFitProvider = self._user_fit_for

    def start(self, invoker: "Invoker") -> None:
        self._invoker = invoker
        user_fit.set_user_fit_provider(self._provider)

    def stop(self, invoker: "Invoker") -> None:
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
    def _calibration_batch(request: WorkingMemoryCalibrationRequest, run_id: str) -> Batch:
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
        # A cached output would skip the very operation the run measures.
        for node in graph.nodes.values():
            node.use_cache = False
        try:
            return Batch(
                graph=graph,
                data=request.data,
                runs=1,
                origin=calibration_origin(run_id),
                destination=CALIBRATION_DESTINATION,
                idempotency_key=request.idempotency_key,
            )
        except ValueError as error:
            raise CalibrationGraphError(str(error)) from error

    @staticmethod
    def _run_id_for(user_id: str, idempotency_key: Optional[str], attempt_id: Optional[str] = None) -> str:
        """A fresh run id, or with a retry key the same one for every repeat of the request: the run id is part of the
        batch's origin, so a replay must rebuild the identical batch for the queue to recognise it."""
        if idempotency_key is None:
            return uuid_string()
        # Without an attempt, the id a key derived before attempts existed; with one, a different id per attempt.
        name = f"{user_id}\x00{idempotency_key}" + (f"\x00{attempt_id}" if attempt_id is not None else "")
        return str(uuid.uuid5(_RUN_ID_NAMESPACE, name))

    async def create_run(
        self, model_key: str, request: WorkingMemoryCalibrationRequest, user_id: str
    ) -> WorkingMemoryCalibrationRunCreated:
        run_id = self._run_id_for(user_id, request.idempotency_key, request.attempt_id)

        def prepare() -> tuple[Batch, str, bool]:
            """The batch, the run's device class, and whether this call recorded the run (a replay finds it)."""
            device = self._calibration_device()
            # Raises UnknownModelException for an unknown key.
            config = self._services.model_manager.store.get_model(model_key)
            batch = self._calibration_batch(request, run_id)
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
                soft=env.soft,
            )

            def add(file: CalibrationStoreFile) -> tuple[bool, bool]:
                if any(known.run_id == run_id for known in file.runs):
                    return False, False
                file.runs.append(run)
                _add_env(file, env)
                return True, True

            created = self._store.change(device_class.id, add, create=CalibrationDeviceClass.of(device_class))
            self._run_files[run_id] = device_class.id
            return batch, device_class.id, bool(created)

        batch, device_class_id, created = await asyncio.to_thread(prepare)
        try:
            result = await self._services.session_queue.enqueue_batch(
                queue_id=DEFAULT_QUEUE_ID, batch=batch, prepend=False, user_id=user_id
            )
        except BaseException:
            # Only a run this call recorded: a conflicting replay must not remove the original request's run.
            if created:
                await asyncio.to_thread(self._discard_run, run_id)
            raise
        if result.enqueued == 0:
            if created:
                await asyncio.to_thread(self._discard_run, run_id)
            raise CalibrationQueueFullError("The queue is full; the calibration run was not started.")

        def remember_items(file: CalibrationStoreFile) -> tuple[bool, None]:
            for run in file.runs:
                if run.run_id == run_id and run.queue_item_ids != result.item_ids:
                    run.queue_item_ids = list(result.item_ids)
                    return True, None
            return False, None

        try:
            await asyncio.to_thread(self._store.change, device_class_id, remember_items)
        except Exception:
            # The items are enqueued and carry the run's origin, which is what the status reads them by.
            self._services.logger.warning(f"Could not record the items of calibration run {run_id}", exc_info=True)
        return WorkingMemoryCalibrationRunCreated(
            run_id=run_id, device_class_id=device_class_id, item_ids=list(result.item_ids)
        )

    def _discard_run(self, run_id: str) -> None:
        try:
            self.reset_run(run_id)
        except Exception:
            self._services.logger.warning(f"Could not discard calibration run {run_id}", exc_info=True)

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
        device_class_id, _ = found
        if device_class_id != self._device_class(device).id:
            return (
                f"Calibration run {run_id} was created for device class {device_class_id}, and this machine's GPU "
                f"is of class {self._device_class(device).id}."
            )
        self._observe(
            run_id,
            queue_item.item_id,
            ObservedItem(
                status="in_progress", retried_from_item_id=queue_item.retried_from_item_id, observed_at=_now()
            ),
        )
        return None

    def _observe(self, run_id: str, item_id: int, observation: ObservedItem) -> None:
        found = self._find_run(run_id)
        if found is None:
            return

        def record(file: CalibrationStoreFile) -> tuple[bool, None]:
            for run in file.runs:
                if run.run_id == run_id:
                    previous = run.observed.get(item_id)
                    if observation.retried_from_item_id is None and previous is not None:
                        observation_ = observation.model_copy(
                            update={"retried_from_item_id": previous.retried_from_item_id}
                        )
                    else:
                        observation_ = observation
                    run.observed[item_id] = observation_
                    return True, None
            return False, None

        self._store.change(found[0], record)

    def observe(self, queue_item: SessionQueueItem, run_id: str, error: Optional[BaseException] = None) -> None:
        try:
            current = self._services.session_queue.get_queue_item(queue_item.item_id)
        except SessionQueueItemNotFoundError:
            # Removed while it ran (Clear Queue, a deleted item): it will not finish.
            self._observe(
                run_id,
                queue_item.item_id,
                ObservedItem(
                    status="canceled",
                    error_message="The queue item was removed from the queue.",
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

    def record_windows(self, run_id: str, windows: list[MeasuredWindow]) -> None:
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
            if not self._store.change(run_class_id, partial(_add_points, run_id=run_id, env=env, points=points)):
                self._services.logger.warning(
                    f"Dropped {len(group)} working-memory measurements of calibration run {run_id}: the run was reset."
                )

    # endregion

    # region Status

    def _live_items(self) -> dict[str, dict[int, QUEUE_ITEM_STATUS]]:
        """The queue's calibration items, per run, as the queue has them now."""
        queue = self._services.session_queue
        item_ids = queue.get_queue_item_ids(DEFAULT_QUEUE_ID, origin_prefix=CALIBRATION_ORIGIN_PREFIX).item_ids
        live: dict[str, dict[int, QUEUE_ITEM_STATUS]] = {}
        for start in range(0, len(item_ids), _SUMMARY_CHUNK):
            for summary in queue.get_queue_item_summaries_by_ids(
                DEFAULT_QUEUE_ID, item_ids[start : start + _SUMMARY_CHUNK]
            ):
                # The queue's origin filter ignores case; only the exact prefix is the server's.
                run_id = calibration_run_id(summary.origin)
                if run_id is not None:
                    live.setdefault(run_id, {})[summary.item_id] = summary.status
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
            if is_current and read.file is None:
                # Creating a run would have to write this file, which is kept as it is.
                unavailable = f"The calibration store of this GPU's device class cannot be written: {read.problem}"
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
        live: Mapping[str, Mapping[int, QUEUE_ITEM_STATUS]],
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
        points_per_env: dict[str, int] = {}
        counts: dict[tuple[str, int, bool, bool], int] = {}
        for point in file.points:
            points_per_run[point.run_id] = points_per_run.get(point.run_id, 0) + 1
            points_per_env[point.env_id] = points_per_env.get(point.env_id, 0) + 1
            key = (point.estimator, point.version, point.valid, point.fit_eligible)
            counts[key] = counts.get(key, 0) + 1
        runs = []
        for run in file.runs:
            point_count = points_per_run.get(run.run_id, 0)
            status, error, item_ids = derive_run_status(run, live.get(run.run_id, {}), point_count)
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
                    item_ids=item_ids,
                    point_count=point_count,
                    attempt_id=run.attempt_id,
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
                settled = known.settled_items if known is not None else frozenset()
                file: Optional[CalibrationStoreFile] = None
                if known is not None and (known.signature, known.env_id, known.platform) == (
                    signature,
                    env.id,
                    platform,
                ):
                    if known.file is None:
                        fits[key] = known
                        continue
                    # The same file: only a run that finished since (in the queue, unwritten) can change the fit.
                    live_runs, settled = self._live_runs(known.file, settled)
                    if live_runs == known.live_runs:
                        fits[key] = replace(known, settled_items=settled)
                        continue
                    file = known.file
                elif signature is not None:
                    read = self._store.read(device_class.id)
                    if read is not None and read.file is None:
                        self._services.logger.warning(
                            f"Working-memory estimates on {device} keep the shipped values: {read.problem}"
                        )
                    file = read.file if read is not None else None
                live_runs, settled = self._live_runs(file, settled) if file is not None else (frozenset(), settled)
                fit = user_fit_of(file, env.id, platform, exclude_runs=live_runs)[0] if file is not None else None
                fits[key] = _DeviceFit(
                    signature=signature,
                    env_id=env.id,
                    platform=platform,
                    fit=fit,
                    file=file,
                    live_runs=live_runs,
                    settled_items=settled,
                )
            self._fits = fits

    def _live_runs(
        self, file: CalibrationStoreFile, settled: frozenset[int] = frozenset()
    ) -> tuple[frozenset[str], frozenset[int]]:
        """The runs whose windows no fit uses yet: those still running (`derive_run_status`; a run is recorded with its
        items when it is enqueued, so one whose items all wait counts), and every run of an attempt (`attempt_id`) one
        of whose runs still runs. And `settled` with the items found finished. A finished queue item never runs again
        (a retry is a new item), so a caller that keeps `settled` asks the queue about each item until it finished,
        and no more.

        Reads the queue, by id, only for the items the file does not record as finished and `settled` does not hold:
        those pending or running, and those that finished without the runner (canceled while pending, cleared). Once
        every run's items finished, it reads nothing. A retry the runner has not started yet is not in the file, so
        it holds back nothing until it starts."""
        open_items: dict[str, list[int]] = {}
        for run in file.runs:
            unfinished = [
                item_id
                for item_id in set(run.queue_item_ids) | set(run.observed)
                if item_id not in settled
                and ((observed := run.observed.get(item_id)) is None or observed.status in _RUNNING)
            ]
            if unfinished:
                open_items[run.run_id] = unfinished
        if not open_items:
            return frozenset(), settled
        item_ids = sorted({item_id for unfinished in open_items.values() for item_id in unfinished})
        statuses: dict[int, QUEUE_ITEM_STATUS] = {}
        queue = self._services.session_queue
        for start in range(0, len(item_ids), _SUMMARY_CHUNK):
            chunk = item_ids[start : start + _SUMMARY_CHUNK]
            for summary in queue.get_queue_item_summaries_by_ids(DEFAULT_QUEUE_ID, chunk):
                statuses[summary.item_id] = summary.status
        live = set()
        for run in file.runs:
            unfinished = open_items.get(run.run_id)
            # Only "running" matters: an item left out reads as finished, and so does a settled one.
            if unfinished is not None:
                queued = {item_id: statuses[item_id] for item_id in unfinished if item_id in statuses}
                if derive_run_status(run, queued, point_count=0)[0] == "running":
                    live.add(run.run_id)
        # A client posts each step of an attempt as its own run, all of them up front: the finished earlier steps
        # must not price the later, larger ones either.
        attempts = {run.attempt_id for run in file.runs if run.run_id in live and run.attempt_id is not None}
        live |= {run.run_id for run in file.runs if run.attempt_id is not None and run.attempt_id in attempts}
        # Gone from the queue counts as finished too: it will not run.
        finished = {item_id for item_id in item_ids if statuses.get(item_id) not in _RUNNING}
        return frozenset(live), settled | finished

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
            reservation=reservation,
            specs=import_estimator_modules(),
        )

    def _present_backends(self) -> dict[str, Backend]:
        """The backend of each device class a GPU generation device here belongs to, by device class id."""
        present: dict[str, Backend] = {}
        for device in self._devices():
            backend = backend_for(device)
            if backend is not None:
                present[self._device_class(device).id] = backend
        return present

    def _present(self, device_class_id: str) -> Optional[tuple[Environment, Platform]]:
        """The environment and platform of a GPU generation device of this class; None when no such device is one."""
        for device in self._devices():
            if backend_for(device) is not None and self._device_class(device).id == device_class_id:
                return self._environment(device), platform_for(device)
        return None

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

        lines, excluded = model_lines(
            file,
            model_key,
            present[0] if present is not None else None,
            platform_of,
            import_estimator_modules(),
            priced_on=decode_lines_of(target, self._services.configuration.precision),
            name_of=name_of,
            live_runs=self._live_runs(file)[0],
        )
        points_per_run: dict[str, int] = {}
        for point in file.points:
            if point.run_id in run_ids:
                points_per_run[point.run_id] = points_per_run.get(point.run_id, 0) + 1
        live = self._live_items() if runs else {}
        model_runs = []
        for run in sorted(runs, key=lambda run: run.created_at, reverse=True):
            point_count = points_per_run.get(run.run_id, 0)
            status, error, _ = derive_run_status(run, live.get(run.run_id, {}), point_count)
            model_runs.append(
                WorkingMemoryCalibrationModelRun(
                    run_id=run.run_id,
                    status=status,
                    error=error,
                    created_at=run.created_at,
                    point_count=point_count,
                    attempt_id=run.attempt_id,
                )
            )
        return summary.model_copy(update={"lines": lines, "excluded_windows": excluded, "runs": model_runs})

    # endregion

    # region Reset and export

    @staticmethod
    def _remove(
        file: CalibrationStoreFile, run_ids: set[str], model_key: Optional[str] = None
    ) -> tuple[bool, WorkingMemoryCalibrationResetResult]:
        """Remove the runs `run_ids` with their points, and with `model_key` every point the summary attributes to
        that model (`measured_models`)."""
        models = run_models(file)
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
        _prune_envs(file)
        return result.runs_removed > 0 or result.points_removed > 0, result

    def reset_run(self, run_id: str) -> WorkingMemoryCalibrationResetResult:
        found = self._find_run(run_id)
        if found is None:
            raise CalibrationRunNotFoundError(run_id)
        result = self._store.change(found[0], lambda file: self._remove(file, {run_id}))
        self._run_files.pop(run_id, None)
        self._try_refresh_user_fits("after a reset")
        return result or WorkingMemoryCalibrationResetResult(runs_removed=0, points_removed=0)

    def reset_model(self, model_key: str, device_class_id: Optional[str] = None) -> WorkingMemoryCalibrationResetResult:
        def remove(file: CalibrationStoreFile) -> tuple[bool, WorkingMemoryCalibrationResetResult]:
            run_ids = {run.run_id for run in file.runs if run.model_key == model_key}
            return self._remove(file, run_ids, model_key)

        result = self._store.change(self._current_device_class_id(device_class_id), remove)
        self._try_refresh_user_fits("after a reset")
        return result or WorkingMemoryCalibrationResetResult(runs_removed=0, points_removed=0)

    def reset_all(self, device_class_id: Optional[str] = None) -> WorkingMemoryCalibrationResetResult:
        def remove(file: CalibrationStoreFile) -> tuple[bool, WorkingMemoryCalibrationResetResult]:
            result = WorkingMemoryCalibrationResetResult(runs_removed=len(file.runs), points_removed=len(file.points))
            changed = bool(file.runs or file.points or file.envs)
            file.runs, file.points, file.envs = [], [], []
            return changed, result

        result = self._store.change(self._current_device_class_id(device_class_id), remove)
        self._try_refresh_user_fits("after a reset")
        return result or WorkingMemoryCalibrationResetResult(runs_removed=0, points_removed=0)

    def export(self, device_class_id: Optional[str] = None) -> Optional[bytes]:
        return self._store.read_raw(self._current_device_class_id(device_class_id))

    # endregion
