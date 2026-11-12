from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class ModelTransferTask:
    remote_url: str
    model_hash: str
    cancel_requested: threading.Event = field(default_factory=threading.Event)
    shared_lock: threading.Lock = field(default_factory=threading.Lock)


_LOCK = threading.Lock()
_TRANSFER_TASKS: dict[int, ModelTransferTask] = {}
_TRANSFER_SEQUENCE = 0


def register_model_transfer(remote_url: str, model_hash: str) -> tuple[int, ModelTransferTask]:
    """Register one generation participating in a possibly shared worker-side model install."""
    global _TRANSFER_SEQUENCE
    task = ModelTransferTask(remote_url=remote_url, model_hash=model_hash)

    with _LOCK:
        task.shared_lock = next(
            (
                active.shared_lock
                for active in _TRANSFER_TASKS.values()
                if active.remote_url == remote_url and active.model_hash == model_hash
            ),
            task.shared_lock,
        )
        _TRANSFER_SEQUENCE += 1
        transfer_id = _TRANSFER_SEQUENCE
        _TRANSFER_TASKS[transfer_id] = task

    return transfer_id, task


def unregister_model_transfer(transfer_id: int) -> None:
    with _LOCK:
        _TRANSFER_TASKS.pop(transfer_id, None)


def another_generation_needs_model(task: ModelTransferTask) -> bool:
    """Preserve a shared install while another live generation still requires it."""
    with _LOCK:
        return any(
            other is not task
            and not other.cancel_requested.is_set()
            and other.remote_url == task.remote_url
            and other.model_hash == task.model_hash
            for other in _TRANSFER_TASKS.values()
        )
