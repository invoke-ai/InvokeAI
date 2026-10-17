"""Working-memory calibration runs: what is stored per device class, and what the API returns about it.

A calibration run is a batch an administrator enqueues to measure one model's working memory. Its queue items carry the
reserved origin `CALIBRATION_ORIGIN_PREFIX` + run id; the session runner measures every node of such an item with the
working-memory probe and hands the windows to the calibration service, which keeps them in one JSON file per device
class (`CalibrationStoreFile`). Recording only: nothing here changes an estimate.
"""

import datetime
from typing import Final, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from invokeai.app.services.session_queue.session_queue_common import QUEUE_ITEM_STATUS, BatchDataCollection
from invokeai.app.services.shared.graph import Graph
from invokeai.backend.working_memory.counters import Backend
from invokeai.backend.working_memory.fingerprint import DeviceClass, Environment
from invokeai.backend.working_memory.probe import MeasuredWindow

CALIBRATION_ORIGIN_PREFIX: Final = "invokeai:wm-calibration:"
"""Queue items whose origin starts with this are calibration items. Only the calibration service may enqueue them."""
CALIBRATION_DESTINATION: Final = "invokeai:wm-calibration"
"""The destination of calibration items: not a gallery or canvas destination, so no client routes their results."""
STORE_SCHEMA_VERSION: Final = 1


def calibration_origin(run_id: str) -> str:
    return CALIBRATION_ORIGIN_PREFIX + run_id


def calibration_run_id(origin: Optional[str]) -> Optional[str]:
    """The run id a queue item's origin names, or None for an item that is not a calibration item."""
    if origin is None or not origin.startswith(CALIBRATION_ORIGIN_PREFIX):
        return None
    return origin[len(CALIBRATION_ORIGIN_PREFIX) :]


def is_reserved_origin(origin: Optional[str]) -> bool:
    """Whether a client-supplied origin claims the calibration prefix. Case-insensitive, because the queue's origin
    filters are: a look-alike origin would otherwise be counted among a run's items."""
    return origin is not None and origin.casefold().startswith(CALIBRATION_ORIGIN_PREFIX)


# region Store
#
# One policy for every store model: unknown fields make the file unusable (`extra="forbid"`). Ignoring them instead
# would drop them on the next rewrite, destroying data a newer writer put there.


class _StoreModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CalibrationDeviceClass(_StoreModel):
    """A `fingerprint.DeviceClass` with its id as a plain field, as it is stored and returned."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(description="A short stable hash of the other fields")
    vendor: str
    name: str
    total_mib: int
    backend: Backend
    arch: Optional[str]

    @classmethod
    def of(cls, device_class: DeviceClass) -> "CalibrationDeviceClass":
        return cls(id=device_class.id, **device_class.model_dump(exclude={"id"}))

    @model_validator(mode="after")
    def _id_matches(self) -> "CalibrationDeviceClass":
        if DeviceClass(**self.model_dump(exclude={"id"})).id != self.id:
            raise ValueError(f"Device class id {self.id} does not match its fields")
        return self


class CalibrationEnvironment(_StoreModel):
    """A `fingerprint.Environment` with its id as a plain field."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(description="A short stable hash of the hard facts")
    hard: dict[str, str]
    soft: dict[str, Optional[str]]

    @classmethod
    def of(cls, env: Environment) -> "CalibrationEnvironment":
        return cls(id=env.id, hard=env.hard, soft=env.soft)

    @model_validator(mode="after")
    def _id_matches(self) -> "CalibrationEnvironment":
        if Environment(hard=self.hard, soft=self.soft).id != self.id:
            raise ValueError(f"Environment id {self.id} does not match its hard facts")
        return self


class CalibrationPoint(MeasuredWindow):
    """One measured window, keyed to the environment it ran under. Forbids unknown fields, as `MeasuredWindow` does."""

    env_id: str


class ObservedItem(_StoreModel):
    """What the session runner saw of one of the run's queue items. Kept so that a run's status survives the queue
    pruning its items."""

    status: QUEUE_ITEM_STATUS
    error_message: Optional[str] = None
    retried_from_item_id: Optional[int] = None
    observed_at: datetime.datetime


class CalibrationRun(_StoreModel):
    run_id: str
    model_key: str
    model_hash: Optional[str] = Field(description="The model's hash when the run was created; provenance only")
    user_id: str
    created_at: datetime.datetime
    env_id: str = Field(description="The environment the run was created under")
    queue_item_ids: list[int] = Field(default_factory=list, description="The items the run was enqueued with")
    observed: dict[int, ObservedItem] = Field(
        default_factory=dict, description="The last status the session runner saw per item, retries included"
    )


class CalibrationStoreFile(_StoreModel):
    """The content of `<db_dir>/working_memory/<device_class_id>.json`, which the export returns as it is."""

    schema_version: Literal[1]
    device_class: CalibrationDeviceClass
    runs: list[CalibrationRun] = Field(default_factory=list)
    points: list[CalibrationPoint] = Field(default_factory=list)
    envs: list[CalibrationEnvironment] = Field(default_factory=list)


# endregion

# region API


class WorkingMemoryCalibrationRequest(BaseModel):
    """The batch a calibration run enqueues, built by the client. The server sets origin, destination and runs, and
    turns the node cache off."""

    model_config = ConfigDict(extra="forbid")

    graph: Graph = Field(description="The graph to measure")
    data: Optional[BatchDataCollection] = Field(default=None, description="The batch data collection")
    idempotency_key: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="A caller-owned retry key. Repeating a request with it returns the run it created and enqueues "
        "nothing new; reusing it with a different request is rejected.",
    )


class WorkingMemoryCalibrationRunCreated(BaseModel):
    run_id: str
    device_class_id: str
    item_ids: list[int] = Field(description="The queue items the run enqueued")


CalibrationRunStatus = Literal["running", "completed", "failed", "partial", "canceled"]


class WorkingMemoryCalibrationRunSummary(BaseModel):
    run_id: str
    model_key: str
    model_hash: Optional[str]
    user_id: str
    created_at: datetime.datetime
    env_id: str
    status: CalibrationRunStatus = Field(
        description="Derived from the run's queue items, a retry standing in for the item it retried: running while "
        "any is pending or running; failed when any failed; completed when all completed; otherwise partial when "
        "points were recorded, else canceled"
    )
    error: Optional[str] = Field(description="The error of a failed item, when the run failed")
    item_ids: list[int] = Field(description="Every queue item of the run, retries included")
    point_count: int


class WorkingMemoryCalibrationPointCount(BaseModel):
    estimator: str
    version: int
    valid: bool
    fit_eligible: bool
    count: int


class WorkingMemoryCalibrationEnvironment(BaseModel):
    id: str
    hard: dict[str, str]
    soft: dict[str, Optional[str]]
    hard_matches_current: Optional[bool] = Field(
        description="Whether the current environment has the same hard facts; null off the current device class"
    )
    soft_matches_current: Optional[bool] = Field(
        description="Whether the current environment has the same soft facts; null off the current device class"
    )
    point_count: int


class WorkingMemoryCalibrationDeviceClass(BaseModel):
    device_class_id: str
    device_class: Optional[CalibrationDeviceClass] = Field(description="Null when the file is unusable")
    usable: bool
    problem: Optional[str] = Field(description="Why the file is unusable. It is kept as it is and never written")
    is_current: bool
    runs: list[WorkingMemoryCalibrationRunSummary]
    point_counts: list[WorkingMemoryCalibrationPointCount]
    envs: list[WorkingMemoryCalibrationEnvironment]


class WorkingMemoryCalibrationStatus(BaseModel):
    available: bool = Field(
        description="Whether a calibration run can start now: one GPU generation device whose store file is usable"
    )
    unavailable_reason: Optional[str]
    current_device_class_id: Optional[str]
    current_env_id: Optional[str]
    device_classes: list[WorkingMemoryCalibrationDeviceClass]


class WorkingMemoryCalibrationResetResult(BaseModel):
    runs_removed: int
    points_removed: int


# endregion

# region Errors


class CalibrationUnavailableError(Exception):
    """This machine cannot calibrate now: no GPU, or more than one generation device."""


class CalibrationGraphError(ValueError):
    """The graph cannot be measured as a calibration run."""


class CalibrationRunNotFoundError(KeyError):
    def __init__(self, run_id: str) -> None:
        super().__init__(run_id)
        self.run_id = run_id

    def __str__(self) -> str:
        return f"Calibration run {self.run_id} does not exist"


class CalibrationQueueFullError(Exception):
    """The queue had no room for the run's items."""


class CalibrationStoreError(Exception):
    """The store could not be read or written."""


class CalibrationStoreUnusableError(CalibrationStoreError):
    """The device class's file has an unknown schema or is invalid; it is kept untouched and not written."""


# endregion
