from __future__ import annotations

from typing import Any

AUTOMATIC_REMOTE_WORKER_NODE_ID = "__irw_remote_worker_dispatch__"
AUTOMATIC_REMOTE_WORKER_NODE_TYPE = "irw_builtin_remote_worker_dispatch"


def schedule_automatic_remote_dispatches(*, batch: Any, item_ids: list[int], services: Any) -> bool:
    """Start the backend worker pool for an automatic Remote Workers batch."""
    helper = batch.graph.nodes.get(AUTOMATIC_REMOTE_WORKER_NODE_ID)
    if helper is None or helper.get_type() != AUTOMATIC_REMOTE_WORKER_NODE_TYPE:
        return False

    from invokeai.app.invocations.remote_worker.worker_pool import schedule_remote_worker_pool

    for item_id in item_ids:
        schedule_remote_worker_pool(int(item_id), services)
    return True
