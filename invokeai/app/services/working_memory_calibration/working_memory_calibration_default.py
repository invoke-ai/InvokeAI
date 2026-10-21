import asyncio
import datetime
import hashlib
import json
import uuid
import weakref
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import ExitStack, asynccontextmanager
from functools import partial
from itertools import chain
from typing import TYPE_CHECKING, Final, NamedTuple, Optional

import torch

from invokeai.app.invocations.call_saved_workflow import CallSavedWorkflowInvocation
from invokeai.app.services.session_queue.session_queue_common import (
    DEFAULT_QUEUE_ID,
    QUEUE_ITEM_STATUS,
    Batch,
    EnqueueIdempotencyConflictError,
    QueueItemLineage,
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
    CalibrationStoreError,
    CalibrationStoreFile,
    CalibrationStoreUnusableError,
    CalibrationUnavailableError,
    ObservedItem,
    WorkingMemoryCalibrationDeviceClass,
    WorkingMemoryCalibrationEnvironment,
    WorkingMemoryCalibrationPointCount,
    WorkingMemoryCalibrationRequest,
    WorkingMemoryCalibrationResetResult,
    WorkingMemoryCalibrationRunCreated,
    WorkingMemoryCalibrationRunSummary,
    WorkingMemoryCalibrationStatus,
    calibration_origin,
    calibration_run_id,
)
from invokeai.app.services.working_memory_calibration.working_memory_calibration_store import (
    CalibrationStore,
    StoreRead,
)
from invokeai.app.util.misc import uuid_string
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.working_memory import fingerprint
from invokeai.backend.working_memory.counters import backend_for
from invokeai.backend.working_memory.fingerprint import DeviceClass, Environment
from invokeai.backend.working_memory.probe import MeasuredWindow

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
    """Attach `points` to their run, except those of a model whose reset epoch moved past `epochs`: a reset removed
    the model's points after the points were measured. How many it attached; none when the file lacks the run."""
    if all(run.run_id != run_id for run in file.runs):
        return False, 0
    current = file.model_reset_epochs
    kept = [point for point in points if current.get(point.model_key, 0) <= epochs.get(point.model_key, 0)]
    if not kept:
        return False, 0
    _add_env(file, env)
    file.points.extend(kept)
    return True, len(kept)


def derive_run_status(
    run: CalibrationRun, live: Mapping[int, QueueItemLineage], valid_point_count: int
) -> tuple[CalibrationRunStatus, Optional[str], list[int]]:
    """A run's status, error and items, from its items: `live` as the queue has them now, the run's own observations
    for items the queue no longer has. An item the queue no longer has counts as canceled unless the runner saw it
    finish. A retry stands in for the item it retried, also before the runner started it. A run that ended without
    completing is partial only with valid points: an invalid one (a canceled or failed operation, a spill) measured
    nothing usable."""
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
    return ("partial" if valid_point_count > 0 else "canceled"), None, item_ids


class _Enqueue(NamedTuple):
    """What a creation enqueues: the batch, the run's device class, whether this request recorded the run, and
    whether it marks the run's enqueue, and so may clear the mark again when nothing was queued."""

    batch: Batch
    device_class_id: str
    created: bool
    marks: bool


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
    ) -> None:
        self._store = store
        self._generation_devices = generation_devices
        self._device_class_for = device_class_for
        self._environment_for = environment_for
        # A card's class cannot change while the process runs; reading it touches the driver.
        self._device_classes: dict[str, DeviceClass] = {}
        # Which file holds a run, so admitting an item and recording its windows do not scan every file.
        self._run_files: dict[str, str] = {}
        # Creations in flight, referenced here so that one whose request was canceled still runs to its end.
        self._creations: set[asyncio.Task[WorkingMemoryCalibrationRunCreated]] = set()
        # Per run id while a creation of this process uses it, in front of the file lock (see `_creation_lock`).
        self._run_gates: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()
        self._invoker: Optional["Invoker"] = None

    def start(self, invoker: "Invoker") -> None:
        self._invoker = invoker

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
        """Check the request's graph and turn its node cache off, then return the run's id, which also becomes the
        graph's id.

        Without a retry key the id is fresh. With one it is derived from the owner, the key and the request's
        fingerprint: the model and the batch the server builds (graph and batch data). A repeat of the request finds
        its run by that id, and another request reusing the key gets another id. The id is part of the batch's
        origin, so the queue's receipt for the key refuses that other request. The client's graph id is not part of
        the request: a graph sent without one gets a fresh id each time, so a repeat could never match.
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
        # A cached output would skip the very operation the run measures.
        for node in graph.nodes.values():
            node.use_cache = False
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
            run_id = str(uuid.uuid5(_RUN_ID_NAMESPACE, f"{user_id}\x00{request.idempotency_key}\x00{digest}"))
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
        gate = self._run_gates.get(run_id)
        if gate is None:
            gate = self._run_gates[run_id] = asyncio.Lock()
        async with gate:
            held = ExitStack()
            # A file lock: taken in a worker thread, which waits for another holder without blocking the loop.
            acquiring = asyncio.ensure_future(asyncio.to_thread(held.enter_context, self._store.creation_lock(run_id)))
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
        # Recording the run, enqueuing its batch and recording its items or removing it: one step for each run id, in
        # every process serving the root.
        async with self._creation_lock(run_id):
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
            # Marked as it is recorded: the enqueue follows, and nothing else is written in between.
            enqueue_attempted=True,
        )

        def add(file: CalibrationStoreFile) -> tuple[bool, bool]:
            if run_id in file.removed_run_ids:
                raise CalibrationRunNotFoundError(run_id, _RESET)
            if any(known.run_id == run_id for known in file.runs):
                return False, False
            file.runs.append(run)
            _add_env(file, env)
            return True, True

        created = bool(self._store.change(device_class.id, add, create=CalibrationDeviceClass.of(device_class)))
        self._run_files[run_id] = device_class.id
        return _Enqueue(batch, device_class.id, created=created, marks=created)

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
                self._remove_run(run_id, remember=False)
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

    def _observe(self, run_id: str, item_id: int, observation: ObservedItem, *, over_completed: bool = True) -> None:
        """Record what the runner saw of the item; without `over_completed`, not over a completion it recorded."""
        found = self._find_run(run_id)
        if found is None:
            return

        def record(file: CalibrationStoreFile) -> tuple[bool, None]:
            for run in file.runs:
                if run.run_id == run_id:
                    previous = run.observed.get(item_id)
                    if not over_completed and previous is not None and previous.status == "completed":
                        return False, None
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
            # Removed while it ran (Clear Queue, a deleted item): it will not finish. Unless the runner recorded its
            # completion: the queue then completed it and pruned it since. A row that still exists has the last word.
            self._observe(
                run_id,
                queue_item.item_id,
                ObservedItem(
                    status="canceled",
                    error_message="The queue item was removed from the queue.",
                    retried_from_item_id=queue_item.retried_from_item_id,
                    observed_at=_now(),
                ),
                over_completed=False,
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

    def observe_completed(self, queue_item: SessionQueueItem, run_id: str) -> None:
        self._observe(
            run_id,
            queue_item.item_id,
            ObservedItem(status="completed", retried_from_item_id=queue_item.retried_from_item_id, observed_at=_now()),
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
                self._device_class_status(read, is_current, current_env if is_current else None, live)
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
    ) -> WorkingMemoryCalibrationDeviceClass:
        file = read.file
        if file is None:
            return WorkingMemoryCalibrationDeviceClass(
                device_class_id=read.device_class_id,
                device_class=None,
                usable=False,
                problem=read.problem,
                is_current=is_current,
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
            status, error, item_ids = derive_run_status(
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
                    item_ids=item_ids,
                    point_count=point_count,
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

    # region Reset and export

    @staticmethod
    def _remove(
        file: CalibrationStoreFile, run_ids: set[str], model_key: Optional[str] = None, *, remember: bool = True
    ) -> tuple[bool, WorkingMemoryCalibrationResetResult]:
        """Remove the runs and their points, and with `model_key` every point of that model. `remember` keeps the
        removed runs' ids, so that a repeat of the request that started one cannot start it again."""
        if remember:
            file.removed_run_ids.extend(
                run.run_id for run in file.runs if run.run_id in run_ids and _is_keyed(run.run_id)
            )
        runs = [run for run in file.runs if run.run_id not in run_ids]
        points = [
            point
            for point in file.points
            if point.run_id not in run_ids and (model_key is None or point.model_key != model_key)
        ]
        result = WorkingMemoryCalibrationResetResult(
            runs_removed=len(file.runs) - len(runs), points_removed=len(file.points) - len(points)
        )
        file.runs, file.points = runs, points
        _prune_envs(file)
        return result.runs_removed > 0 or result.points_removed > 0, result

    def reset_run(self, run_id: str) -> WorkingMemoryCalibrationResetResult:
        return self._remove_run(run_id, remember=True)

    def _remove_run(self, run_id: str, *, remember: bool) -> WorkingMemoryCalibrationResetResult:
        found = self._find_run(run_id)
        if found is None:
            raise CalibrationRunNotFoundError(run_id)
        result = self._store.change(found[0], lambda file: self._remove(file, {run_id}, remember=remember))
        self._run_files.pop(run_id, None)
        return result or WorkingMemoryCalibrationResetResult(runs_removed=0, points_removed=0)

    def reset_model(self, model_key: str, device_class_id: Optional[str] = None) -> WorkingMemoryCalibrationResetResult:
        class_id = self._current_device_class_id(device_class_id)

        def remove(file: CalibrationStoreFile) -> tuple[bool, WorkingMemoryCalibrationResetResult]:
            # Written with the removal or not at all: a node that read the older epoch drops its points of the model.
            file.model_reset_epochs[model_key] = file.model_reset_epochs.get(model_key, 0) + 1
            run_ids = {run.run_id for run in file.runs if run.model_key == model_key}
            return True, self._remove(file, run_ids, model_key)[1]

        result = self._store.change(class_id, remove)
        return result or WorkingMemoryCalibrationResetResult(runs_removed=0, points_removed=0)

    def reset_all(self, device_class_id: Optional[str] = None) -> WorkingMemoryCalibrationResetResult:
        def remove(file: CalibrationStoreFile) -> tuple[bool, WorkingMemoryCalibrationResetResult]:
            result = WorkingMemoryCalibrationResetResult(runs_removed=len(file.runs), points_removed=len(file.points))
            changed = bool(file.runs or file.points or file.envs)
            file.removed_run_ids.extend(run.run_id for run in file.runs if _is_keyed(run.run_id))
            file.runs, file.points, file.envs = [], [], []
            return changed, result

        result = self._store.change(self._current_device_class_id(device_class_id), remove)
        return result or WorkingMemoryCalibrationResetResult(runs_removed=0, points_removed=0)

    def export(self, device_class_id: Optional[str] = None) -> Optional[bytes]:
        return self._store.read_raw(self._current_device_class_id(device_class_id))

    # endregion
