"""Working-memory calibration runs: what is stored per device class, and what the API returns about it.

A calibration run is a batch an administrator enqueues to measure one model's working memory. Its queue items carry the
reserved origin `CALIBRATION_ORIGIN_PREFIX` + run id; the session runner measures every node of such an item with the
working-memory probe and hands the windows to the calibration service, which keeps them in one JSON file per device
class (`CalibrationStoreFile`). The service fits the windows of the current environment into user lines that estimates
read (`backend/working_memory/user_fit.py`), plans the cases a client runs to calibrate a model, and summarises what a
model's runs measured.
"""

import datetime
from typing import Final, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, model_serializer, model_validator

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
ATTEMPT_STEPS_MAX: Final = 256
"""The most runs one attempt may plan. A plan has at most a few dozen steps (`working_memory_calibration_plan.py`)."""


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
    enqueue_attempted: bool = Field(
        default=False,
        description="Set before the run's batch is enqueued, and cleared only by the attempt that set it, when its "
        "enqueue queued nothing. A repeat of the request that finds no items of the run then enqueues nothing: they "
        "may have been queued",
    )
    observed: dict[int, ObservedItem] = Field(
        default_factory=dict, description="The last status the session runner saw per item, retries included"
    )
    attempt_id: Optional[str] = Field(default=None, description="The client's attempt the run belongs to, if any")
    attempt_steps: Optional[int] = Field(
        default=None,
        ge=1,
        description="How many runs the attempt planned, as its runs sent it; only with attempt_id. Absent in runs "
        "stored before it was sent, whose attempt is judged on the runs it recorded",
    )
    soft: Optional[dict[str, Optional[str]]] = Field(
        default=None,
        description="The soft facts of the environment the run was created under; absent in runs stored before they "
        "were recorded, which fall back to the environment's",
    )

    @model_validator(mode="after")
    def _steps_belong_to_an_attempt(self) -> "CalibrationRun":
        if self.attempt_steps is not None and self.attempt_id is None:
            raise ValueError(f"Run {self.run_id} plans attempt_steps without an attempt_id")
        return self

    @model_serializer(mode="wrap")
    def _omit_absent_fields(self, handler: SerializerFunctionWrapHandler) -> dict:
        # Written only when set: a run without them stays readable by a version that forbids the unknown fields.
        data = handler(self)
        for name in ("attempt_id", "attempt_steps", "soft"):
            if data.get(name) is None:
                data.pop(name, None)
        return data


class RemovedAttemptSteps(_StoreModel):
    """What resets removed of one attempt (attempt_id): what judging the runs of it that are left needs."""

    removed: int = Field(ge=1, description="How many of the attempt's runs resets removed; they count as queued")
    aborted: bool = Field(
        default=False,
        description="Whether the attempt was running or already aborted when a reset removed a run of it: it then "
        "never feeds a fit",
    )
    attempt_steps: Optional[int] = Field(
        default=None,
        ge=1,
        description="The attempt_steps the removed runs sent, which a later run of the attempt must send too; null "
        "for runs stored before it was sent",
    )


class CalibrationStoreFile(_StoreModel):
    """The content of `<db_dir>/working_memory/<device_class_id>.json`, which the export returns as it is."""

    schema_version: Literal[1]
    device_class: CalibrationDeviceClass
    runs: list[CalibrationRun] = Field(default_factory=list)
    points: list[CalibrationPoint] = Field(default_factory=list)
    envs: list[CalibrationEnvironment] = Field(default_factory=list)
    removed_run_ids: list[str] = Field(
        default_factory=list,
        description="The runs with a retry key that a reset removed, kept for good. A repeat of the request that "
        "started one cannot start it again, so its queue items keep failing at admission",
    )
    model_reset_epochs: dict[str, int] = Field(
        default_factory=dict,
        description="Per model key, how often the model was reset. A node's points of a model reset after the node "
        "started are dropped",
    )
    removed_attempt_steps: dict[str, RemovedAttemptSteps] = Field(
        default_factory=dict,
        description="Per attempt id, what resets removed of the attempt, so that the runs of it that are left are "
        "judged on every run it queued. Kept while a stored run of the attempt is left, or a later step of it may "
        "still come: the attempt was running when a reset removed a run of it",
    )

    @model_serializer(mode="wrap")
    def _omit_absent_fields(self, handler: SerializerFunctionWrapHandler) -> dict:
        # Written only while a reset's record is kept: a file without one stays readable by a version that forbids the
        # unknown field.
        data = handler(self)
        if not data.get("removed_attempt_steps"):
            data.pop("removed_attempt_steps", None)
        return data


# endregion

# region API

ATTEMPT_ID_DESCRIPTION: Final = (
    "A client-generated id of the calibration attempt the run belongs to (1-64 letters, digits, '_', '-', '.', ':'), "
    "so a client that enqueues one run per step can group them. Sent together with attempt_steps, never alone (422). No "
    "window of an attempt feeds an estimate until it queued attempt_steps runs and every one has finished; a run a "
    "reset removed still counts as queued, without its measurements. An aborted attempt never feeds one: an item of "
    "it ended canceled (Stop, a queue clear), it ended before it queued all its runs (Stop, a failed submission), or "
    "a reset removed a run of it while it ran. An item that failed ends that step, not the attempt. "
    "The run id derives from the idempotency_key, the request and the attempt, so an idempotency_key names one run: "
    "repeating the request under the same attempt returns that run, and reusing the key under another attempt is "
    "refused with 409 while the queue keeps the key's receipt. Make each key unique per run of an attempt."
)
ATTEMPT_STEPS_DESCRIPTION: Final = (
    f"How many runs the attempt (attempt_id) queues, 1-{ATTEMPT_STEPS_MAX}. Sent together with attempt_id, never "
    "alone (422), and the same by every run of the attempt: a run that sends another number than the attempt's runs "
    "already stored is refused with 409. The attempt's windows feed estimates only once it queued that many runs, "
    "those a reset removed since included, and all of them have finished with no item canceled."
)
MEASUREMENT_METHOD_DESCRIPTION: Final = (
    "Where a window on a GPU of this class here reads its driver figure (`counters.Method`): torch-only when this "
    "platform has no driver reader; null when no generation device of this class is present"
)
MEASUREMENT_VERIFIED_DESCRIPTION: Final = (
    "Whether that method was checked against real hardware on this backend (`counters.VERIFIED_METHODS`); null when "
    "no generation device of this class is present"
)


class WorkingMemoryCalibrationRequest(BaseModel):
    """The batch a calibration run enqueues, built by the client. The server sets origin, destination, runs and the
    graph's id, and turns the node cache off."""

    model_config = ConfigDict(extra="forbid")

    graph: Graph = Field(description="The graph to measure")
    data: Optional[BatchDataCollection] = Field(default=None, description="The batch data collection")
    idempotency_key: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="A caller-owned retry key. Repeating the request with it returns the run it created and enqueues "
        "nothing new; once a reset removed that run, the repeat is refused. Reusing the key for another request "
        "(another model, graph or batch data) is rejected while the queue keeps the key's enqueue receipt.",
    )
    attempt_id: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_.:-]+$",
        description=ATTEMPT_ID_DESCRIPTION,
    )
    attempt_steps: Optional[int] = Field(
        default=None, ge=1, le=ATTEMPT_STEPS_MAX, description=ATTEMPT_STEPS_DESCRIPTION
    )

    @model_validator(mode="after")
    def _attempt_comes_with_its_steps(self) -> "WorkingMemoryCalibrationRequest":
        if (self.attempt_id is None) != (self.attempt_steps is None):
            raise ValueError("attempt_id and attempt_steps are sent together or not at all")
        return self


class WorkingMemoryCalibrationRunCreated(BaseModel):
    run_id: str
    device_class_id: str
    item_ids: list[int] = Field(
        description="The queue items the run enqueued. A repeat of the request that finds them unrecorded takes over "
        "every queue item carrying the run's origin, retries included; none when the queue no longer has them and the "
        "runner never saw them"
    )


CalibrationRunStatus = Literal["running", "completed", "failed", "partial", "canceled"]

CalibrationHoldBack = Literal["running", "aborted_attempt"]
"""Why no window of a run feeds a fit: running, the queue holds an unfinished item of it or of a run of its attempt;
aborted_attempt, none is running and its attempt was aborted: an item of it ended canceled (Stop, a queue clear), it
queued fewer runs than its attempt_steps (Stop, or a failed submission, ended it before it queued them all; a run a
reset removed counts as queued), or a reset removed a run of it while it ran or after it was aborted. An aborted
attempt never feeds a fit."""

_RUN_ATTEMPT_STEPS_DESCRIPTION: Final = (
    "How many runs the run's attempt planned; null without an attempt, and for a run stored before it was sent"
)


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
        "valid points were recorded, else canceled"
    )
    error: Optional[str] = Field(description="The error of a failed item, when the run failed")
    item_ids: list[int] = Field(description="Every queue item of the run known, retries and the items they retried")
    point_count: int
    attempt_id: Optional[str] = Field(description="The client's attempt the run belongs to; null when none was sent")
    attempt_steps: Optional[int] = Field(description=_RUN_ATTEMPT_STEPS_DESCRIPTION)


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
        description="Whether the current environment has the soft facts first recorded with this environment (`soft`); "
        "a later run's own soft facts are not compared here. Null off the current device class"
    )
    point_count: int


class WorkingMemoryCalibrationDeviceClass(BaseModel):
    device_class_id: str
    device_class: Optional[CalibrationDeviceClass] = Field(description="Null when the file is unusable")
    usable: bool
    problem: Optional[str] = Field(description="Why the file is unusable. It is kept as it is and never written")
    is_current: bool
    measurement_method: Optional[str] = Field(description=MEASUREMENT_METHOD_DESCRIPTION)
    measurement_verified: Optional[bool] = Field(description=MEASUREMENT_VERIFIED_DESCRIPTION)
    runs: list[WorkingMemoryCalibrationRunSummary]
    point_counts: list[WorkingMemoryCalibrationPointCount]
    envs: list[WorkingMemoryCalibrationEnvironment]


class WorkingMemoryCalibrationStatus(BaseModel):
    available: bool = Field(
        description="Whether a calibration run can start now: one GPU generation device whose store file is usable "
        "and can be written"
    )
    unavailable_reason: Optional[str]
    current_device_class_id: Optional[str]
    current_env_id: Optional[str]
    device_classes: list[WorkingMemoryCalibrationDeviceClass]


class WorkingMemoryCalibrationResetResult(BaseModel):
    runs_removed: int
    points_removed: int


class WorkingMemoryCalibrationDeviceClassRef(BaseModel):
    id: str
    name: str


class WorkingMemoryCalibrationCase(BaseModel):
    """One generation a calibration plan runs, in the fields of the client's graph-export case (mode txt2img)."""

    id: str = Field(description="Unique within the plan; becomes the graph id")
    mode: Literal["txt2img"] = "txt2img"
    model_key: str = Field(description="The main model the case generates with")
    components: dict[str, str] = Field(
        description="Component picks by Generate slot key; `vae` when a standalone VAE is calibrated or a GPU VAE "
        "must be picked over one that runs on the CPU"
    )
    width: int
    height: int
    steps: int = Field(description="The fewest steps that still run the whole pipeline")
    seed: int
    positive_prompt: str
    negative_prompt: str
    settings: dict[str, str] = Field(
        description="Generate settings overrides: `vaePrecision` (fp16 or fp32) where the decode node casts the VAE"
    )
    estimator: str = Field(description="The working-memory estimator of the operation the case measures")
    line: str = Field(description="The estimator line the case measures, `<operation>_<dtype>`")


class WorkingMemoryCalibrationRepetitions(BaseModel):
    warm_up: int = Field(description="Runs of each case before the measured ones; their windows count too")
    measured: int


class WorkingMemoryCalibrationSharedModel(BaseModel):
    key: str
    name: str
    base: str
    type: str
    lines: list[str] = Field(description="The lines it shares with the planned model, `<operation>_<dtype>`")


class WorkingMemoryCalibrationPlan(BaseModel):
    """What calibrating one model on this machine's GPU runs, or why it cannot."""

    model_key: str
    supported: bool
    reason: Optional[str] = Field(description="Why the model cannot be calibrated here, when it cannot")
    device_class: Optional[WorkingMemoryCalibrationDeviceClassRef] = Field(
        description="The class of the GPU the cases run on; null without exactly one GPU generation device"
    )
    calibrated_through: Optional[str] = Field(
        description="For a standalone VAE, the installed main model the cases generate with"
    )
    estimators: list[str] = Field(description="The working-memory estimators of the model's covered operations")
    cases: list[WorkingMemoryCalibrationCase]
    repetitions: WorkingMemoryCalibrationRepetitions
    estimated_seconds: Optional[int] = Field(
        description="A rough order of magnitude for all cases and repetitions, model load included; not a measurement"
    )
    shared_with: list[WorkingMemoryCalibrationSharedModel] = Field(
        description="Other installed models whose operations price with the same lines on this GPU, so a calibration "
        "changes their estimates too"
    )
    not_covered: list[str] = Field(description="Operations of the model a calibration does not measure, and why")


CalibrationLineState = Literal["in_use", "stale", "unusable", "insufficient"]

CalibrationSizeExclusion = Literal["fragmented", "partial_residency", "aborted_attempt"]


class WorkingMemoryCalibrationLineSize(BaseModel):
    width: int
    height: int
    pixels: int
    windows: int = Field(
        description="Counted: the windows at this size that feed the line, valid, fit-eligible and not fragmented, of "
        "every model in the environment; those of a calibration run still running are counted, and join the fit once "
        "it finished. Not counted: the valid windows measured at this size"
    )
    measured_bytes: int = Field(description="The largest demand of those windows")
    counted: bool = Field(description="Whether windows of this size feed the user line")
    excluded_reason: Optional[CalibrationSizeExclusion] = Field(
        description="Why a size is not counted: fragmented, its windows measured the allocator's fragmentation "
        "(`backend/working_memory/fragmentation.py`); partial_residency, the model was not fully on the device; "
        "aborted_attempt, its windows come from aborted attempts (Stop, a queue clear, a failed submission), which "
        "never feed a line. Null for a counted size"
    )
    shipped_bytes: Optional[int] = Field(
        description="The shipped estimate at this size, priced with the line's estimator where one of its windows "
        "measured the size, otherwise with that of the largest window; null when the platform of the environment is "
        "unknown"
    )
    user_bytes: Optional[int] = Field(
        description="The estimate with the user line applied, as estimates read it (without the windows of runs still "
        "running); null without a user line"
    )


class WorkingMemoryCalibrationPixelRange(BaseModel):
    min_pixels: int
    max_pixels: int


class WorkingMemoryCalibrationLineContributor(BaseModel):
    """A model whose windows feed a line: the model a run calibrated, or the model a window measured."""

    model_key: str
    name: Optional[str] = Field(description="The model's name; null when it is no longer installed")
    windows: int = Field(description="Its counted windows on the line: valid, fit-eligible and not fragmented")


class WorkingMemoryCalibrationModelLine(BaseModel):
    """One estimator line in one environment: one the model's windows measured, or, in the current environment, one
    the model's decode prices on that other models' windows measured."""

    estimator: str
    version: int
    line: str = Field(description="`<operation>_<dtype>`")
    carrier: Optional[str] = Field(description="The estimator whose shipped entry carries the line on this platform")
    env_id: str
    state: CalibrationLineState = Field(
        description="in_use: the user line applies; stale: it applies, but the soft facts of the newest finished run "
        "that contributed a window differ from the current ones; unusable: hard facts "
        "differ from the current environment (or no GPU of this class generates here), so shipped values apply; "
        "insufficient: the windows of finished, complete attempts cannot fit a line (yet); `problem` says why"
    )
    changed_facts: list[str] = Field(
        description="The soft facts that changed (stale) or the hard facts that differ (unusable)"
    )
    problem: Optional[str] = Field(description="Why no line could be fitted, when it could not")
    measured_range: Optional[WorkingMemoryCalibrationPixelRange] = Field(
        description="The sizes the user line was fitted on, over every model's windows of the line from finished runs"
    )
    slope_bytes_per_pixel: Optional[int]
    base_bytes: Optional[int]
    sizes: list[WorkingMemoryCalibrationLineSize]
    own_windows: int = Field(description="The model's own windows that feed the line; 0 when only other models' do")
    measured_by: list[WorkingMemoryCalibrationLineContributor] = Field(
        description="Every model whose windows feed the line, this one included when it has any; by name, then key"
    )


class WorkingMemoryCalibrationModelRun(BaseModel):
    run_id: str
    status: CalibrationRunStatus
    error: Optional[str]
    created_at: datetime.datetime
    point_count: int
    attempt_id: Optional[str] = Field(description="The client's attempt the run belongs to; null when none was sent")
    attempt_steps: Optional[int] = Field(description=_RUN_ATTEMPT_STEPS_DESCRIPTION)
    held_back: Optional[CalibrationHoldBack] = Field(
        description="Why no window of the run feeds a fit: running, it or a run of its attempt is still in the queue, "
        "and its windows join once all have finished; aborted_attempt, its attempt was aborted, and its windows never "
        "feed one: an item of it ended canceled (Stop, a queue clear), the attempt ended before it queued all "
        "attempt_steps runs (Stop, a failed submission), or a reset removed a run of it while it ran. Null when its "
        "windows are judged like any finished run's"
    )


class WorkingMemoryCalibrationModelSummary(BaseModel):
    """What one model's calibration runs measured on one device class, and what estimates make of it."""

    model_key: str
    device_class_id: str
    usable: bool
    problem: Optional[str] = Field(description="Why the store file is unusable")
    current_env_id: Optional[str] = Field(description="Null when no generation device of this class is present")
    measurement_method: Optional[str] = Field(description=MEASUREMENT_METHOD_DESCRIPTION)
    measurement_verified: Optional[bool] = Field(description=MEASUREMENT_VERIFIED_DESCRIPTION)
    lines: list[WorkingMemoryCalibrationModelLine]
    excluded_windows: int = Field(
        description="The model's windows that feed no line: invalid, not fit-eligible, fragmented, of an aborted "
        "attempt, tiled, batched, upcast, a VAE outside the AutoencoderKL family, no demand, or of an estimator a user "
        "calibration does not move, such as the Z-Image denoise. Of these, only the not fit-eligible, the fragmented "
        "and the aborted attempt's windows show among a line's sizes, as a size that is not counted where no counted "
        "window measured it"
    )
    runs: list[WorkingMemoryCalibrationModelRun]


# endregion

# region Errors


class CalibrationUnavailableError(Exception):
    """This machine cannot calibrate now: no GPU, or more than one generation device."""


class CalibrationGraphError(ValueError):
    """The graph cannot be measured as a calibration run."""


class CalibrationRunNotFoundError(KeyError):
    def __init__(self, run_id: str, reason: str = "does not exist") -> None:
        super().__init__(run_id)
        self.run_id = run_id
        self.reason = reason

    def __str__(self) -> str:
        return f"Calibration run {self.run_id} {self.reason}"


class CalibrationQueueFullError(Exception):
    """The queue had no room for all of the run's items, so it took none."""


class CalibrationAttemptConflictError(Exception):
    """The run's attempt_steps differs from that of the run, or of the attempt's other runs, already stored."""


class CalibrationStoreError(Exception):
    """The store could not be read or written."""


class CalibrationStoreUnusableError(CalibrationStoreError):
    """The device class's file has an unknown schema or is invalid; it is kept untouched and not written."""


# endregion
