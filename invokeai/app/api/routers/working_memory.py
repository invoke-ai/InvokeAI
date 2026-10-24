"""Working-memory calibration runs: administrators measure what a model's operations really take on this machine's GPU.

Mounted under the model manager's prefix: a run calibrates one model (`/i/{key}/working_memory/...`), and its
measurements belong to the machine's device class (`/working_memory/...`). Every route is admin-only: runs spend GPU
time on the instance's behalf, and the store is machine data that names the accounts that created runs.
"""

import asyncio
from typing import Final, Optional

from fastapi import Body, HTTPException, Path, Query, Response
from fastapi.routing import APIRouter

from invokeai.app.api.auth_dependencies import AdminUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api.routers.image_move_maintenance import assert_image_move_maintenance_inactive
from invokeai.app.services.model_records import UnknownModelException
from invokeai.app.services.session_queue.session_queue_common import (
    EnqueueIdempotencyConflictError,
    EnqueueReceiptLimitError,
)
from invokeai.app.services.working_memory_calibration.working_memory_calibration_base import (
    WorkingMemoryCalibrationServiceBase,
)
from invokeai.app.services.working_memory_calibration.working_memory_calibration_common import (
    CalibrationAttemptConflictError,
    CalibrationGraphError,
    CalibrationQueueFullError,
    CalibrationRunNotFoundError,
    CalibrationStoreError,
    CalibrationStoreUnusableError,
    CalibrationUnavailableError,
    WorkingMemoryCalibrationModelSummary,
    WorkingMemoryCalibrationPlan,
    WorkingMemoryCalibrationRequest,
    WorkingMemoryCalibrationResetResult,
    WorkingMemoryCalibrationRunCreated,
    WorkingMemoryCalibrationStatus,
)

working_memory_router = APIRouter(prefix="/v2/models", tags=["model_manager"])

_DEVICE_CLASS_ID = Query(
    default=None,
    pattern=r"^[0-9a-f]{16}$",
    description="The device class to act on; the current generation device's by default",
)


# One description per error, shared by the routes that return it.
_STORE_UNREADABLE: Final = {
    "description": "The calibration store cannot be read or written (unwritable root, lock timeout)"
}
_NO_DEVICE_CLASS: Final = {
    "description": "No device_class_id was given and there is no current one (no GPU, or several generation "
    "devices), or the device class's store file is unusable and kept as it is"
}


def _service() -> WorkingMemoryCalibrationServiceBase:
    service = ApiDependencies.invoker.services.working_memory_calibration
    if service is None:
        raise HTTPException(status_code=503, detail="Working-memory calibration is not available")
    return service


def _http_error(error: Exception) -> HTTPException:
    if isinstance(error, (CalibrationRunNotFoundError, UnknownModelException)):
        return HTTPException(status_code=404, detail=str(error))
    if isinstance(error, CalibrationGraphError):
        return HTTPException(status_code=422, detail=str(error))
    if isinstance(
        error,
        (
            CalibrationUnavailableError,
            CalibrationStoreUnusableError,
            CalibrationQueueFullError,
            CalibrationAttemptConflictError,
            EnqueueIdempotencyConflictError,
        ),
    ):
        return HTTPException(status_code=409, detail=str(error))
    if isinstance(error, EnqueueReceiptLimitError):
        return HTTPException(status_code=429, detail=str(error))
    # CalibrationStoreError: the root cannot be read or written (read-only, lock timeout). Also the 503 of `_service`.
    return HTTPException(status_code=503, detail=str(error))


@working_memory_router.post(
    "/i/{key}/working_memory/calibrate",
    operation_id="calibrate_model_working_memory",
    status_code=201,
    response_model=WorkingMemoryCalibrationRunCreated,
    responses={
        404: {
            "description": "The model could not be found; the run was removed while it was being created, whose "
            "items are queued then and fail when they start; or a reset removed the run the idempotency key started"
        },
        409: {
            "description": "This machine cannot calibrate now (no GPU, or several generation devices), the device "
            "class's store file is unusable and kept as it is (possibly after the items were queued, which then fail "
            "when they start), the queue has no room for all of the run's items (it then takes none), the "
            "idempotency key was used for another request, or attempt_steps differs from that of the attempt's runs "
            "already stored (or of the run a repeated request names)"
        },
        422: {
            "description": "The graph cannot be measured as a calibration run, or attempt_id and attempt_steps were "
            "not sent together"
        },
        429: {"description": "Too many unacknowledged idempotency keys; acknowledge or wait, then retry"},
        503: {
            "description": "The calibration store cannot be read or written (unwritable root, lock timeout); with an "
            "idempotency key, possibly after the items were queued"
        },
    },
)
async def calibrate_model_working_memory(
    current_admin: AdminUserOrDefault,
    key: str = Path(description="Key of the model the run calibrates"),
    request: WorkingMemoryCalibrationRequest = Body(description="The batch to measure"),
) -> WorkingMemoryCalibrationRunCreated:
    """Start a working-memory calibration run: enqueue the batch with every node uncached and measure each node's
    model operations on the generation device. Needs exactly one generation device, and it must be a GPU. The queue
    takes all of the batch's items or none.

    Once the items are queued, a request without an `idempotency_key` succeeds even when the store fails for a passing
    reason (an unwritable root, a lock timeout) to record them: they run all the same, and the status finds them by
    their origin. With a key, that failure is a 503 although the items are queued. A run removed meanwhile (404) or an
    unusable store file (409) fails the request with or without a key; its queued items then fail when they start.

    With a key, repeating the request (also after a lost response) records the queued items and returns the run; it
    enqueues nothing new. The items come from the queue's enqueue receipt for the key while it exists, otherwise from
    the queue items that carry the run's origin and those the runner saw. Once a reset removed the run, a repeat is
    refused (404). Acknowledge the key through the queue's enqueue_batch/acknowledge route once the response
    arrived."""
    await asyncio.to_thread(assert_image_move_maintenance_inactive)
    try:
        return await _service().create_run(key, request, current_admin.user_id)
    except (
        UnknownModelException,
        CalibrationRunNotFoundError,
        CalibrationGraphError,
        CalibrationUnavailableError,
        CalibrationStoreError,
        CalibrationQueueFullError,
        CalibrationAttemptConflictError,
        EnqueueIdempotencyConflictError,
        EnqueueReceiptLimitError,
    ) as error:
        raise _http_error(error) from error


@working_memory_router.get(
    "/working_memory/status",
    operation_id="get_working_memory_calibration_status",
    response_model=WorkingMemoryCalibrationStatus,
    responses={503: _STORE_UNREADABLE},
)
def get_working_memory_calibration_status(current_admin: AdminUserOrDefault) -> WorkingMemoryCalibrationStatus:
    """Every device class's calibration runs, with their derived status, point counts and environments."""
    try:
        return _service().get_status()
    except CalibrationStoreError as error:
        raise _http_error(error) from error


@working_memory_router.get(
    "/i/{key}/working_memory/plan",
    operation_id="get_model_working_memory_calibration_plan",
    response_model=WorkingMemoryCalibrationPlan,
    responses={404: {"description": "The model could not be found"}},
)
def get_model_working_memory_calibration_plan(
    current_admin: AdminUserOrDefault,
    key: str = Path(description="Key of the model to calibrate"),
) -> WorkingMemoryCalibrationPlan:
    """The text-to-image cases that calibrate the model's decode on this machine's GPU, how often each runs, which
    other installed models the calibration also moves, or why the model cannot be calibrated here (`supported`)."""
    try:
        return _service().get_plan(key)
    except UnknownModelException as error:
        raise _http_error(error) from error


@working_memory_router.get(
    "/i/{key}/working_memory/summary",
    operation_id="get_model_working_memory_calibration_summary",
    response_model=WorkingMemoryCalibrationModelSummary,
    responses={
        404: {"description": "The model could not be found"},
        409: {"description": "No device_class_id was given and there is no current one"},
        503: _STORE_UNREADABLE,
    },
)
def get_model_working_memory_calibration_summary(
    current_admin: AdminUserOrDefault,
    key: str = Path(description="Key of the model whose measurements to summarise"),
    device_class_id: Optional[str] = _DEVICE_CLASS_ID,
) -> WorkingMemoryCalibrationModelSummary:
    """What the calibration windows of the model's lines measured on one device class, per estimator line and
    environment: the sizes, the largest demand, the shipped and the user estimate, whether the user line is in use,
    stale or unusable, and which models measured it, also for a line the model's decode shares with other models'
    measurements; and the model's runs."""
    try:
        return _service().get_model_summary(key, device_class_id)
    except (UnknownModelException, CalibrationUnavailableError, CalibrationStoreError) as error:
        raise _http_error(error) from error


@working_memory_router.delete(
    "/working_memory/runs/{run_id}",
    operation_id="delete_working_memory_calibration_run",
    response_model=WorkingMemoryCalibrationResetResult,
    responses={
        404: {"description": "No usable store file holds the run"},
        409: {"description": "The run's store file became unusable after the run was found; it is kept as it is"},
        503: _STORE_UNREADABLE,
    },
)
def delete_working_memory_calibration_run(
    current_admin: AdminUserOrDefault,
    run_id: str = Path(description="The calibration run to remove"),
) -> WorkingMemoryCalibrationResetResult:
    """Remove a calibration run and its measurements. Its queue items that have not run yet fail when they start.

    A removed run of an attempt still counts toward the attempt's runs (`attempt_steps`), so the rest of a finished
    attempt is judged as before. Removing a run while its attempt runs, or after it was stopped (a canceled item),
    stops the attempt for good: none of its measurements count."""
    try:
        return _service().reset_run(run_id)
    except (CalibrationRunNotFoundError, CalibrationStoreError) as error:
        raise _http_error(error) from error


@working_memory_router.delete(
    "/i/{key}/working_memory",
    operation_id="reset_model_working_memory_calibration",
    response_model=WorkingMemoryCalibrationResetResult,
    responses={409: _NO_DEVICE_CLASS, 503: _STORE_UNREADABLE},
)
def reset_model_working_memory_calibration(
    current_admin: AdminUserOrDefault,
    key: str = Path(description="Key of the model whose runs and measurements to remove"),
    device_class_id: Optional[str] = _DEVICE_CLASS_ID,
) -> WorkingMemoryCalibrationResetResult:
    """Remove a model's calibration runs with all their measurements, and every measurement of the model taken in
    other runs (a VAE measured inside a transformer's run, say), from one device class. Removed runs of an attempt
    still count toward its runs; an attempt that was running is stopped for good, its later steps included."""
    try:
        return _service().reset_model(key, device_class_id)
    except (CalibrationUnavailableError, CalibrationStoreError) as error:
        raise _http_error(error) from error


@working_memory_router.delete(
    "/working_memory",
    operation_id="reset_working_memory_calibration",
    response_model=WorkingMemoryCalibrationResetResult,
    responses={409: _NO_DEVICE_CLASS, 503: _STORE_UNREADABLE},
)
def reset_working_memory_calibration(
    current_admin: AdminUserOrDefault,
    device_class_id: Optional[str] = _DEVICE_CLASS_ID,
) -> WorkingMemoryCalibrationResetResult:
    """Remove every calibration run and measurement of one device class. An attempt that was running is stopped
    for good: its later steps never count either."""
    try:
        return _service().reset_all(device_class_id)
    except (CalibrationUnavailableError, CalibrationStoreError) as error:
        raise _http_error(error) from error


@working_memory_router.get(
    "/working_memory/export",
    operation_id="export_working_memory_calibration",
    response_class=Response,
    responses={
        200: {
            "description": "The device class's store file byte for byte: a CalibrationStoreFile (schema_version 1) "
            "when this version can read it, otherwise whatever the file holds",
            "content": {"application/json": {}},
        },
        404: {"description": "The device class has no store file"},
        409: {"description": "No device_class_id was given and there is no current one"},
        503: _STORE_UNREADABLE,
    },
)
def export_working_memory_calibration(
    current_admin: AdminUserOrDefault,
    device_class_id: Optional[str] = _DEVICE_CLASS_ID,
) -> Response:
    """The device class's calibration store file as it is on disk, readable by this version or not."""
    try:
        raw = _service().export(device_class_id)
    except (CalibrationUnavailableError, CalibrationStoreError) as error:
        raise _http_error(error) from error
    if raw is None:
        raise HTTPException(status_code=404, detail="This device class has no calibration store")
    return Response(content=raw, media_type="application/json")
