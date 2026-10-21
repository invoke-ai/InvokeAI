from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Optional

from invokeai.app.services.session_queue.session_queue_common import SessionQueueItem
from invokeai.app.services.working_memory_calibration.working_memory_calibration_common import (
    WorkingMemoryCalibrationModelSummary,
    WorkingMemoryCalibrationPlan,
    WorkingMemoryCalibrationRequest,
    WorkingMemoryCalibrationResetResult,
    WorkingMemoryCalibrationRunCreated,
    WorkingMemoryCalibrationStatus,
)
from invokeai.backend.working_memory.probe import MeasuredWindow


class WorkingMemoryCalibrationServiceBase(ABC):
    """Calibration runs: enqueues them, admits their queue items, and keeps their measurements per device class."""

    @abstractmethod
    async def create_run(
        self, model_key: str, request: WorkingMemoryCalibrationRequest, user_id: str
    ) -> WorkingMemoryCalibrationRunCreated:
        """Validate the request, record the run and enqueue its batch as `user_id`, who must be an administrator.

        The queue takes all of the batch's items or none (`CalibrationQueueFullError`). With a retry key, a repeat of
        the request returns the run it created and enqueues nothing; once a reset removed the run, it raises
        `CalibrationRunNotFoundError`."""

    @abstractmethod
    def admit(self, queue_item: SessionQueueItem, run_id: str) -> Optional[str]:
        """Called by the session runner before a calibration item runs: None to run it, or why it must fail."""

    @abstractmethod
    def model_reset_epochs(self, run_id: str) -> dict[str, int]:
        """The reset epoch of every model reset on the run's device class; empty when the run does not exist. Read
        when a node starts and passed to `record_windows`."""

    @abstractmethod
    def record_windows(
        self, run_id: str, windows: list[MeasuredWindow], model_reset_epochs: Optional[Mapping[str, int]] = None
    ) -> None:
        """Persist one node's measured windows. Windows of a run that no longer exists are dropped, and so are those
        of a model reset since `model_reset_epochs` was read (by default, at this call): the reset removed the
        model's points, these among them. A reset by another process sharing the root counts as well."""

    @abstractmethod
    def observe(self, queue_item: SessionQueueItem, run_id: str, error: Optional[BaseException] = None) -> None:
        """Record the item's queue status in its run, after the session runner is done with it. `error` is what the
        runner raised, which fails an item the queue still has as running; a removed item counts as canceled."""

    @abstractmethod
    def get_status(self) -> WorkingMemoryCalibrationStatus:
        """Every device class's runs, with derived status, point counts and environments."""

    @abstractmethod
    def reset_run(self, run_id: str) -> WorkingMemoryCalibrationResetResult:
        """Remove a run and its points. Every reset keeps the ids of the runs it removed: a repeat of the request that
        started one cannot start it again."""

    @abstractmethod
    def reset_model(self, model_key: str, device_class_id: Optional[str] = None) -> WorkingMemoryCalibrationResetResult:
        """Remove a model's runs and points from one device class, the current one by default."""

    @abstractmethod
    def reset_all(self, device_class_id: Optional[str] = None) -> WorkingMemoryCalibrationResetResult:
        """Remove every run and point of one device class, the current one by default."""

    @abstractmethod
    def export(self, device_class_id: Optional[str] = None) -> Optional[bytes]:
        """The device class's store file as it is on disk, the current one by default; None when it has none."""

    @abstractmethod
    def refresh_user_fits(self) -> None:
        """Bring the user fits estimates read (`backend/working_memory/user_fit.py`) up to date with the store: per
        GPU generation device, the fit of its device class's windows in the current environment, leaving out the
        windows of calibration runs still running. Re-reads a file only when its signature or the environment changed,
        so an unchanged store whose runs all finished costs one stat per device; while a run is unfinished, also one
        queue read of its unfinished items by id. The session runner calls this when it starts any queue item;
        building a plan and a reset call it too. Estimates read the fits from memory."""

    @abstractmethod
    def get_plan(self, model_key: str) -> WorkingMemoryCalibrationPlan:
        """The cases that calibrate the model on this machine's GPU, or why it cannot be calibrated here. Raises
        UnknownModelException for an unknown key."""

    @abstractmethod
    def get_model_summary(
        self, model_key: str, device_class_id: Optional[str] = None
    ) -> WorkingMemoryCalibrationModelSummary:
        """What the windows that feed the model's lines measured on one device class, the current one by default: per
        line and environment, the shipped and the user estimate at each size, whether the user line applies, and which
        models measured it; and the model's runs. Raises UnknownModelException for an unknown key."""
