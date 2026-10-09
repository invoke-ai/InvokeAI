from __future__ import annotations

import json
import re
import threading
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

from invokeai.app.invocations.primitives import ImageOutput, VideoOutput
from invokeai.app.invocations.remote_worker.remote_client import RemoteInvokeError
from invokeai.app.services.shared.invocation_context import InvocationContextData, build_invocation_context

_AUTOMATIC_NODE_ID = "__irw_remote_worker_dispatch__"
_AUTOMATIC_NODE_TYPE = "irw_builtin_remote_worker_dispatch"

_POOL_LOCK = threading.Lock()
_POOLS: dict[tuple[str, str], "_RemoteWorkerPool"] = {}
_SLOT_LOCKS_GUARD = threading.Lock()
_SLOT_LOCKS: dict[str, threading.Lock] = {}
_LOCAL_HANDOFFS_GUARD = threading.Lock()
_LOCAL_HANDOFF_ITEMS: set[int] = set()


@dataclass(frozen=True)
class WorkerSpec:
    url: str
    name: str
    slot: int


@dataclass(frozen=True)
class PoolSettings:
    mode: Literal["Distributed", "Remote Only"]
    workers: tuple[WorkerSpec, ...]
    result_destination: Literal["gallery", "canvas"]
    local_gallery_board_id: str
    keep_remote_copies: bool
    auto_transfer_missing_models: bool
    model_transfer_host: str
    model_transfer_timeout_seconds: int
    poll_interval_seconds: float
    timeout_seconds: int


def _node_value(node: Any, name: str, default: Any = None) -> Any:
    if isinstance(node, dict):
        return node.get(name, default)
    return getattr(node, name, default)


def _node_type(node: Any) -> str:
    if isinstance(node, dict):
        return str(node.get("type") or "")
    get_type = getattr(node, "get_type", None)
    if callable(get_type):
        try:
            return str(get_type())
        except Exception:
            pass
    return str(getattr(node, "type", "") or "")


def _helper_for_queue_item(queue_item: Any) -> Any | None:
    try:
        nodes = queue_item.session.graph.nodes
    except Exception:
        return None
    if not isinstance(nodes, dict):
        return None
    direct = nodes.get(_AUTOMATIC_NODE_ID)
    if direct is not None and _node_type(direct) == _AUTOMATIC_NODE_TYPE:
        return direct
    for node in nodes.values():
        if _node_type(node) == _AUTOMATIC_NODE_TYPE:
            return node
    return None


def _split_urls(primary: str, additional: str) -> list[str]:
    raw: list[str] = []
    if primary.strip():
        raw.append(primary.strip())
    raw.extend(part.strip() for part in re.split(r"[,;\n\r]+", additional or "") if part.strip())

    seen: set[str] = set()
    result: list[str] = []
    for value in raw:
        normalized = value.rstrip("/")
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
    return result


def _settings_from_helper(helper: Any) -> PoolSettings | None:
    mode = str(_node_value(helper, "dispatch_mode", "Distributed"))
    if mode not in {"Distributed", "Remote Only"}:
        return None

    urls = _split_urls(
        str(_node_value(helper, "remote_url", "") or ""),
        str(_node_value(helper, "additional_remote_urls", "") or ""),
    )
    if not urls:
        return None

    try:
        decoded = json.loads(str(_node_value(helper, "remote_worker_names", "[]") or "[]"))
    except Exception:
        decoded = []
    names = decoded if isinstance(decoded, list) else []

    workers = tuple(
        WorkerSpec(
            url=url,
            name=(
                str(names[index]).strip() if index < len(names) and str(names[index]).strip() else f"Remote {index + 1}"
            ),
            slot=index + 1,
        )
        for index, url in enumerate(urls)
    )

    destination = str(_node_value(helper, "result_destination", "gallery"))
    if destination not in {"gallery", "canvas"}:
        destination = "gallery"

    return PoolSettings(
        mode=mode,  # type: ignore[arg-type]
        workers=workers,
        result_destination=destination,  # type: ignore[arg-type]
        local_gallery_board_id=str(_node_value(helper, "local_gallery_board_id", "") or ""),
        keep_remote_copies=bool(_node_value(helper, "keep_remote_copies", False)),
        auto_transfer_missing_models=bool(_node_value(helper, "auto_transfer_missing_models", True)),
        model_transfer_host=str(_node_value(helper, "model_transfer_host", "") or ""),
        model_transfer_timeout_seconds=max(
            60, int(_node_value(helper, "model_transfer_timeout_seconds", 7200) or 7200)
        ),
        poll_interval_seconds=max(0.25, float(_node_value(helper, "collector_poll_interval_seconds", 0.75) or 0.75)),
        timeout_seconds=max(10, int(_node_value(helper, "collector_timeout_seconds", 14400) or 14400)),
    )


def _settings_for_queue_item(queue_item: Any) -> PoolSettings | None:
    helper = _helper_for_queue_item(queue_item)
    return _settings_from_helper(helper) if helper is not None else None


def _worker_allowed(settings: PoolSettings, worker: WorkerSpec) -> bool:
    return any(candidate.url.casefold() == worker.url.casefold() for candidate in settings.workers)


def _slot_lock(worker: WorkerSpec) -> threading.Lock:
    key = worker.url.rstrip("/").casefold()
    with _SLOT_LOCKS_GUARD:
        lock = _SLOT_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _SLOT_LOCKS[key] = lock
        return lock


def _register_local_handoff(item_id: int) -> None:
    with _LOCAL_HANDOFFS_GUARD:
        _LOCAL_HANDOFF_ITEMS.add(int(item_id))


def _unregister_local_handoff(item_id: int) -> None:
    with _LOCAL_HANDOFFS_GUARD:
        _LOCAL_HANDOFF_ITEMS.discard(int(item_id))


def _is_local_handoff(item_id: int) -> bool:
    with _LOCAL_HANDOFFS_GUARD:
        return int(item_id) in _LOCAL_HANDOFF_ITEMS


def _queue_item_ids(
    services: Any,
    *,
    queue_id: str,
    user_id: str,
    statuses: tuple[str, ...],
) -> list[int]:
    if not statuses:
        return []

    db = getattr(services.session_queue, "_db", None)
    if db is None:
        raise RuntimeError("Remote worker pool requires InvokeAI's SQLite session queue")

    placeholders = ", ".join("?" for _ in statuses)
    with db.transaction() as cursor:
        cursor.execute(
            f"""--sql
            SELECT item_id
            FROM session_queue
            WHERE queue_id = ?
              AND user_id = ?
              AND status IN ({placeholders})
              AND parent_item_id IS NULL
            ORDER BY priority DESC, item_id ASC
            """,
            (queue_id, user_id, *statuses),
        )
        return [int(row[0]) for row in cursor.fetchall()]


def _status(services: Any, item_id: int) -> str:
    try:
        return str(services.session_queue.get_queue_item(item_id).status or "").lower()
    except Exception:
        return ""


def _park_remote_only_items(services: Any, queue_id: str, user_id: str) -> None:
    queue = services.session_queue
    dequeue_lock = getattr(queue, "_dequeue_lock", None)
    set_status = getattr(queue, "_set_queue_item_status", None)
    if dequeue_lock is None or set_status is None:
        raise RuntimeError("Remote worker pool requires queue claim primitives")

    for item_id in _queue_item_ids(
        services,
        queue_id=queue_id,
        user_id=user_id,
        statuses=("pending",),
    ):
        try:
            item = queue.get_queue_item(item_id)
        except Exception:
            continue

        settings = _settings_for_queue_item(item)
        if settings is None or settings.mode != "Remote Only":
            continue

        with dequeue_lock:
            try:
                fresh = queue.get_queue_item(item_id)
            except Exception:
                continue
            fresh_settings = _settings_for_queue_item(fresh)
            if fresh.status != "pending" or fresh_settings is None or fresh_settings.mode != "Remote Only":
                continue
            set_status(item_id=item_id, status="waiting", queue_item=fresh)


def _claim_for_worker(
    services: Any,
    *,
    queue_id: str,
    user_id: str,
    worker: WorkerSpec,
    excluded: set[int],
) -> tuple[Any, PoolSettings] | None:
    queue = services.session_queue
    dequeue_lock = getattr(queue, "_dequeue_lock", None)
    set_status = getattr(queue, "_set_queue_item_status", None)
    if dequeue_lock is None or set_status is None:
        raise RuntimeError("Remote worker pool requires queue claim primitives")

    for item_id in _queue_item_ids(
        services,
        queue_id=queue_id,
        user_id=user_id,
        statuses=("pending", "waiting"),
    ):
        if _is_local_handoff(item_id):
            continue
        if item_id in excluded:
            continue

        try:
            candidate = queue.get_queue_item(item_id)
        except Exception:
            continue
        settings = _settings_for_queue_item(candidate)
        if settings is None or not _worker_allowed(settings, worker):
            continue
        if settings.mode == "Distributed" and candidate.status != "pending":
            continue
        if settings.mode == "Remote Only" and candidate.status not in {"pending", "waiting"}:
            continue

        with dequeue_lock:
            try:
                fresh = queue.get_queue_item(item_id)
            except Exception:
                continue
            fresh_settings = _settings_for_queue_item(fresh)
            if fresh_settings is None or not _worker_allowed(fresh_settings, worker):
                continue
            if fresh_settings.mode == "Distributed" and fresh.status != "pending":
                continue
            if fresh_settings.mode == "Remote Only" and fresh.status not in {"pending", "waiting"}:
                continue

            claimed = set_status(
                item_id=item_id,
                status="in_progress",
                device=f"remote:{worker.name}",
                queue_item=fresh,
            )
            if claimed.status == "in_progress":
                return claimed, fresh_settings

    return None


def _has_eligible_item(
    services: Any,
    *,
    queue_id: str,
    user_id: str,
    worker: WorkerSpec,
) -> bool:
    for item_id in _queue_item_ids(
        services,
        queue_id=queue_id,
        user_id=user_id,
        statuses=("pending", "waiting"),
    ):
        try:
            item = services.session_queue.get_queue_item(item_id)
        except Exception:
            continue
        settings = _settings_for_queue_item(item)
        if settings is not None and _worker_allowed(settings, worker):
            return True
    return False


def _requeue(services: Any, item_id: int, settings: PoolSettings, reason: str) -> None:
    if _status(services, item_id) != "in_progress":
        return

    target = "waiting" if settings.mode == "Remote Only" else "pending"
    set_status = getattr(services.session_queue, "_set_queue_item_status", None)
    if set_status is None:
        raise RuntimeError("Remote worker pool requires queue status transitions")
    set_status(item_id=item_id, status=target)
    services.logger.warning(f"Remote Workers: returned local item {item_id} to {target}: {reason}")


def _is_remote_oom_error(errors: Any) -> bool:
    """Return True only for clear remote device-memory exhaustion failures."""
    try:
        detail = json.dumps(errors, ensure_ascii=False).casefold()
    except Exception:
        detail = str(errors).casefold()
    return "outofmemoryerror" in detail or "cuda out of memory" in detail or "would exceed allowed memory" in detail


def _fail_remote_oom(services: Any, queue_item: Any, worker: WorkerSpec, errors: Any) -> None:
    item_id = int(queue_item.item_id)
    if _status(services, item_id) not in {"in_progress", "waiting", "pending"}:
        return

    detail = json.dumps(errors, ensure_ascii=False)[:4000]
    services.session_queue.fail_queue_item(
        item_id=item_id,
        error_type="OutOfMemoryError",
        error_message=f"{worker.name} ran out of memory: {detail}",
        error_traceback="",
    )
    services.logger.warning(
        f"Remote Workers [{worker.name}]: remote item failed with an out-of-memory error; "
        f"marked local item {item_id} failed instead of requeueing it"
    )


def _event_item(queue_item: Any, invocation: Any) -> Any:
    item = queue_item.model_copy(deep=True)
    invocation_id = str(invocation.id)
    item.session.prepared_source_mapping.setdefault(invocation_id, invocation_id)
    return item


def _emit_started(services: Any, queue_item: Any, invocation: Any, worker: WorkerSpec) -> Any:
    item = _event_item(queue_item, invocation)
    services.events.emit_invocation_started(queue_item=item, invocation=invocation)
    services.events.emit_invocation_progress(
        queue_item=item,
        invocation=invocation,
        message=f"{worker.name} · Preparing",
        percentage=None,
        image=None,
    )
    return item


def _emit_progress(
    services: Any,
    queue_item: Any,
    invocation: Any,
    worker: WorkerSpec,
    *,
    message: str,
    percentage: float | None = None,
    image: Any = None,
    revision: int | None = None,
) -> None:
    safe = str(message or "Rendering").replace("\n", " ").strip()
    services.events.emit_invocation_progress(
        queue_item=queue_item,
        invocation=invocation,
        message=f"{worker.name} · {safe}",
        percentage=percentage,
        image=image,
        revision=revision,
    )


def _emit_result(
    services: Any,
    queue_item: Any,
    event_item: Any,
    invocation: Any,
    source_id: str,
    output: Any,
) -> None:
    synthetic_id = str(uuid.uuid4())
    synthetic = invocation.model_copy(update={"id": synthetic_id})
    source_id = str(source_id or invocation.id)

    # Persist remote results under the REAL source node id. Do not add the
    # synthetic event id to prepared_source_mapping: GraphExecutionState requires
    # every key in that mapping to exist in execution_graph.
    #
    # If the same source produces more than one imported output, keep the prior
    # value under an unmapped synthetic result key and leave the newest/final
    # output at source_id. Unfiltered history still retains both, while consumers
    # filtering for canvas_output/video_output naturally get the final result.
    previous = queue_item.session.results.get(source_id)
    if previous is not None:
        queue_item.session.results[synthetic_id] = previous
    queue_item.session.results[source_id] = output

    # event_item is an ephemeral deep copy used only for live events. Mapping the
    # synthetic event invocation here preserves live source routing without ever
    # persisting a fake execution node.
    event_item.session.prepared_source_mapping[synthetic_id] = source_id
    event_item.session.results[synthetic_id] = output

    services.events.emit_invocation_started(queue_item=event_item, invocation=synthetic)
    services.events.emit_invocation_complete(queue_item=event_item, invocation=synthetic, output=output)


def _persist_remote_results(
    services: Any,
    queue_item: Any,
    worker: WorkerSpec,
    *,
    phase: str,
) -> None:
    try:
        services.session_queue.save_queue_item_session(int(queue_item.item_id), queue_item.session)
    except Exception as exc:
        services.logger.warning(
            f"Remote Workers [{worker.name}]: completed item {queue_item.item_id}, "
            f"but could not persist imported result history {phase}: {exc}"
        )


def _capture_local_output_board(graph: dict[str, Any]) -> str | None:
    """Capture only board assignments from nodes that save visible output media."""
    nodes = graph.get("nodes")
    if not isinstance(nodes, dict):
        return None

    for node in nodes.values():
        if not isinstance(node, dict) or node.get("is_intermediate") is not False:
            continue

        board = node.get("board")
        if isinstance(board, dict):
            value = board.get("board_id") or board.get("id")
            if isinstance(value, str) and value.strip():
                return value.strip()

        value = node.get("board_id")
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def _build_remote_graph(
    queue_item: Any,
    settings: PoolSettings,
    services: Any,
) -> tuple[dict[str, Any], str]:
    from invokeai.app.invocations.remote_worker.model_transfer import enrich_model_identifier_hashes
    from invokeai.app.invocations.remote_worker.remote_nodes import (
        _disable_graph_cache,
        _strip_helper_nodes,
        _strip_remote_board_assignments,
    )

    try:
        dumped = queue_item.model_dump(mode="json")
        source_graph = dumped["session"]["graph"]
    except Exception as exc:
        raise RemoteInvokeError("Could not read executable graph from queue item") from exc

    if not isinstance(source_graph, dict) or not isinstance(source_graph.get("nodes"), dict):
        raise RemoteInvokeError("Queue item did not contain a usable executable graph")

    remote_graph = deepcopy(source_graph)
    remote_graph["id"] = str(uuid.uuid4())

    explicit_board = _capture_local_output_board(source_graph)
    _strip_helper_nodes(remote_graph)
    _strip_remote_board_assignments(remote_graph)
    _disable_graph_cache(remote_graph)
    enrich_model_identifier_hashes(remote_graph, services)

    board_id = explicit_board or settings.local_gallery_board_id.strip()
    if board_id.lower() == "none" or settings.result_destination == "canvas":
        board_id = ""

    return remote_graph, board_id


def _build_context(services: Any, queue_item: Any, invocation: Any):
    item_id = int(queue_item.item_id)

    def canceled() -> bool:
        return _status(services, item_id) in {"canceled", "cancelled", "failed", "completed"}

    return build_invocation_context(
        services=services,
        data=InvocationContextData(
            queue_item=queue_item,
            invocation=invocation,
            source_invocation_id=invocation.id,
        ),
        is_canceled=canceled,
    )


def _cleanup_remote_inputs(
    client: Any,
    services: Any,
    settings: PoolSettings,
    image_names: list[str],
    video_names: list[str],
    *,
    reason: str,
) -> None:
    if settings.keep_remote_copies or (not image_names and not video_names):
        return

    from invokeai.app.invocations.remote_worker.remote_media import cleanup_remote_media

    cleanup_remote_media(
        client=client,
        image_names=list(dict.fromkeys(image_names)),
        video_names=list(dict.fromkeys(video_names)),
        services=services,
        reason=reason,
    )


def _dispatch_remote(
    services: Any,
    queue_item: Any,
    invocation: Any,
    settings: PoolSettings,
    worker: WorkerSpec,
) -> tuple[Any, int, str, list[str], list[str]]:
    from invokeai.app.invocations.remote_worker.model_transfer import resolve_local_model_file
    from invokeai.app.invocations.remote_worker.remote_nodes import (
        RemoteModelTransferCancelled,
        _graph_media_references,
        _remote_client,
        _remote_model_matches_local,
        _transfer_missing_model_to_remote,
        _transfer_source_images_to_remote,
        _transfer_source_videos_to_remote,
    )

    context = _build_context(services, queue_item, invocation)
    graph, board_id = _build_remote_graph(queue_item, settings, services)
    client = _remote_client(worker.url, str(queue_item.user_id))

    # Probe reachability/authentication before expensive preparation.
    client.get_current_item()

    missing_handler = None
    if settings.auto_transfer_missing_models:

        def missing_handler(identifier: dict[str, Any]) -> None:
            _transfer_missing_model_to_remote(
                context=context,
                invocation=invocation,
                remote_client=client,
                remote_index=worker.slot,
                identifier=identifier,
                transfer_host=settings.model_transfer_host,
                timeout_seconds=settings.model_transfer_timeout_seconds,
            )

    local_models: dict[str, Any] = {}

    def model_match_validator(identifier: dict[str, Any], candidate: dict[str, Any]) -> bool:
        local_key = str(identifier.get("key") or "")
        model = local_models.get(local_key)
        if model is None:
            model = resolve_local_model_file(context._services, identifier)
            local_models[local_key] = model
        return _remote_model_matches_local(client, model, candidate)

    try:
        client.remap_model_identifiers(
            graph,
            missing_model_handler=missing_handler,
            model_match_validator=model_match_validator,
        )
    except RemoteModelTransferCancelled:
        raise

    media_refs = _graph_media_references(graph)
    image_names = [ref[6:] for ref in media_refs if ref.startswith("image:")]
    video_names = [ref[6:] for ref in media_refs if ref.startswith("video:")]
    uploaded_image_names: list[str] = []
    uploaded_video_names: list[str] = []

    try:
        if image_names:
            _transfer_source_images_to_remote(
                context=context,
                remote_client=client,
                graph=graph,
                image_names=image_names,
                remote_index=worker.slot,
                uploaded_names=uploaded_image_names,
            )

        if video_names:
            _transfer_source_videos_to_remote(
                context=context,
                remote_client=client,
                graph=graph,
                video_names=video_names,
                remote_index=worker.slot,
                uploaded_names=uploaded_video_names,
            )

        origin = str(getattr(queue_item, "origin", "") or "invokeai")
        remote_item_id = client.enqueue_graph(
            graph=graph,
            queue_id="default",
            origin=f"{origin}:remote-worker:{worker.slot}",
        )
    except Exception:
        _cleanup_remote_inputs(
            client,
            services,
            settings,
            uploaded_image_names,
            uploaded_video_names,
            reason="remote dispatch failed before queueing",
        )
        raise

    return client, int(remote_item_id), board_id, uploaded_image_names, uploaded_video_names


def _remote_media_source_ids(completed_item: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    """Map worker media names back to their original graph source node ids."""
    session = completed_item.get("session")
    if not isinstance(session, dict):
        return {}, {}

    results = session.get("results")
    if not isinstance(results, dict):
        return {}, {}

    raw_mapping = session.get("prepared_source_mapping")
    prepared_source_mapping = raw_mapping if isinstance(raw_mapping, dict) else {}
    image_sources: dict[str, str] = {}
    video_sources: dict[str, str] = {}

    for result_id, result in results.items():
        if not isinstance(result, dict):
            continue

        result_key = str(result_id)
        mapped = prepared_source_mapping.get(result_key)
        source_id = str(mapped) if isinstance(mapped, str) and mapped else result_key

        image = result.get("image")
        if isinstance(image, dict):
            image_name = image.get("image_name")
            if isinstance(image_name, str) and image_name:
                image_sources[image_name] = source_id

        images = result.get("images")
        if isinstance(images, list):
            for entry in images:
                if not isinstance(entry, dict):
                    continue
                image_name = entry.get("image_name")
                if isinstance(image_name, str) and image_name:
                    image_sources[image_name] = source_id

        video_candidates: list[Any] = [result.get("video")]
        videos = result.get("videos")
        if isinstance(videos, list):
            video_candidates.extend(videos)
        video_candidates.append(result)

        for entry in video_candidates:
            if not isinstance(entry, dict):
                continue
            video_name = entry.get("video_name")
            if isinstance(video_name, str) and video_name:
                video_sources[video_name] = source_id

    return image_sources, video_sources


def _import_completed(
    services: Any,
    queue_item: Any,
    invocation: Any,
    settings: PoolSettings,
    client: Any,
    completed_item: dict[str, Any],
    board_id: str,
) -> tuple[list[Any], list[Any]]:
    from invokeai.app.invocations.remote_worker.remote_media import (
        cleanup_remote_media,
        save_local_image,
        save_local_video,
    )

    image_source_ids, video_source_ids = _remote_media_source_ids(completed_item)
    all_images = client.extract_image_names(
        completed_item,
        non_intermediate_only=False,
        allow_empty=True,
    )
    all_videos = client.extract_video_names(completed_item)
    images = client.filter_gallery_image_names(all_images)
    videos = client.filter_gallery_video_names(all_videos)

    if settings.result_destination == "canvas":
        if not images and all_images:
            images = [all_images[-1]]
        if not videos and all_videos:
            videos = [all_videos[-1]]

    if not images and not videos:
        raise RemoteInvokeError("Remote render completed without an importable image or video output")

    image_dtos: list[Any] = []
    video_dtos: list[Any] = []

    for remote_name in images:
        image_metadata = client.get_image_metadata(remote_name)
        image = client.download_image(remote_name)
        image_dtos.append(
            save_local_image(
                services=services,
                queue_item=queue_item,
                invocation=invocation,
                image=image,
                metadata=image_metadata,
                board_id=board_id,
                result_destination=settings.result_destination,
                source_node_id=image_source_ids.get(remote_name),
            )
        )

    for remote_name in videos:
        video_metadata = client.get_video_metadata(remote_name)
        video_bytes = client.download_video(remote_name)
        video_dtos.append(
            save_local_video(
                services=services,
                queue_item=queue_item,
                invocation=invocation,
                video_bytes=video_bytes,
                metadata=video_metadata,
                board_id=board_id,
                result_destination=settings.result_destination,
                source_node_id=video_source_ids.get(remote_name),
            )
        )

    if not settings.keep_remote_copies:
        cleanup_remote_media(
            client=client,
            image_names=all_images,
            video_names=all_videos,
            services=services,
            reason="local import succeeded",
        )

    return image_dtos, video_dtos


def _cancel_remote(
    client: Any,
    remote_item_id: int,
    services: Any,
    worker: WorkerSpec,
) -> bool:
    try:
        client.cancel_queue_item(remote_item_id, "default")
    except Exception as exc:
        services.logger.warning(f"Remote Workers [{worker.name}]: cancellation request failed: {exc}")

    for _ in range(61):
        try:
            item = client.get_item(remote_item_id, "default")
        except Exception:
            return False

        status = str(item.get("status") or "").lower()
        if status in {"completed", "canceled", "cancelled", "failed"}:
            if status != "failed":
                try:
                    client.delete_queue_item(remote_item_id, "default")
                except Exception as exc:
                    services.logger.warning(
                        f"Remote Workers [{worker.name}]: could not delete canceled remote queue item "
                        f"{remote_item_id}: {exc}"
                    )
            return True

        time.sleep(0.5)

    return False


def _run_remote_job(
    services: Any,
    queue_item: Any,
    settings: PoolSettings,
    worker: WorkerSpec,
    *,
    complete_local: bool,
) -> str:
    invocation = _helper_for_queue_item(queue_item)
    if invocation is None:
        raise RemoteInvokeError("Remote worker helper node was not found")

    client, remote_item_id, board_id, input_image_names, input_video_names = _dispatch_remote(
        services,
        queue_item,
        invocation,
        settings,
        worker,
    )
    event_item = _emit_started(services, queue_item, invocation, worker)

    services.logger.info(
        f"Remote Workers [{worker.name}]: local item {queue_item.item_id} -> remote item {remote_item_id}"
    )

    started = time.monotonic()
    last_signature: tuple[Any, Any, Any] | None = None
    last_network_log = 0.0

    while True:
        local_status = _status(services, int(queue_item.item_id))
        if local_status in {"canceled", "cancelled"}:
            remote_stopped = _cancel_remote(client, remote_item_id, services, worker)
            if remote_stopped:
                _cleanup_remote_inputs(
                    client,
                    services,
                    settings,
                    input_image_names,
                    input_video_names,
                    reason="remote job canceled",
                )
            return "canceled"

        if time.monotonic() - started > settings.timeout_seconds:
            remote_stopped = _cancel_remote(client, remote_item_id, services, worker)
            if remote_stopped:
                _cleanup_remote_inputs(
                    client,
                    services,
                    settings,
                    input_image_names,
                    input_video_names,
                    reason="remote job timed out",
                )
            raise RemoteInvokeError(
                f"{worker.name} remote item {remote_item_id} timed out after {settings.timeout_seconds:g}s"
            )

        try:
            item = client.get_item(remote_item_id, "default")
        except Exception as exc:
            now = time.monotonic()
            if now - last_network_log >= 10:
                services.logger.warning(
                    f"Remote Workers [{worker.name}]: temporarily unavailable while monitoring "
                    f"item {remote_item_id}; will retry: {exc}"
                )
                last_network_log = now
            time.sleep(settings.poll_interval_seconds)
            continue

        remote_status = str(item.get("status") or "").lower()

        if remote_status == "completed":
            try:
                image_dtos, video_dtos = _import_completed(
                    services,
                    queue_item,
                    invocation,
                    settings,
                    client,
                    item,
                    board_id,
                )
            finally:
                _cleanup_remote_inputs(
                    client,
                    services,
                    settings,
                    input_image_names,
                    input_video_names,
                    reason="remote job completed",
                )

            for dto in image_dtos:
                _emit_result(
                    services,
                    queue_item,
                    event_item,
                    invocation,
                    source_id=str(getattr(dto, "node_id", None) or invocation.id),
                    output=ImageOutput.build(dto),
                )
            for dto in video_dtos:
                _emit_result(
                    services,
                    queue_item,
                    event_item,
                    invocation,
                    source_id=str(getattr(dto, "node_id", None) or invocation.id),
                    output=VideoOutput.build(dto),
                )

            _emit_progress(
                services,
                event_item,
                invocation,
                worker,
                message="Complete",
                percentage=1.0,
            )

            try:
                client.delete_queue_item(remote_item_id, "default")
            except Exception as exc:
                services.logger.warning(
                    f"Remote Workers [{worker.name}]: imported result but could not delete "
                    f"remote queue item {remote_item_id}: {exc}"
                )

            if complete_local and _status(services, int(queue_item.item_id)) == "in_progress":
                # Persist before the terminal event: the frontend immediately fetches this
                # queue row on completion and filters results by prepared_source_mapping.
                _persist_remote_results(services, queue_item, worker, phase="before completion")
                services.session_queue.complete_queue_item(int(queue_item.item_id))
                # Keep the previous post-completion write as a best-effort safeguard.
                _persist_remote_results(services, queue_item, worker, phase="after completion")

            return "completed"

        if remote_status in {"failed", "canceled", "cancelled"}:
            _cleanup_remote_inputs(
                client,
                services,
                settings,
                input_image_names,
                input_video_names,
                reason=f"remote job ended with status {remote_status}",
            )
            errors = item.get("session", {}).get("errors", {}) if isinstance(item.get("session"), dict) else {}
            if remote_status == "failed" and _is_remote_oom_error(errors):
                _fail_remote_oom(services, queue_item, worker, errors)
            raise RemoteInvokeError(
                f"{worker.name} remote item {remote_item_id} ended with status "
                f"'{remote_status}': {json.dumps(errors)[:2000]}"
            )

        if remote_status in {"in_progress", "running"}:
            try:
                preview = client.get_progress_preview(remote_item_id, "default")
            except Exception:
                preview = None

            if preview:
                signature = (
                    preview.get("revision"),
                    preview.get("percentage"),
                    preview.get("message"),
                )
                if signature != last_signature:
                    from invokeai.app.invocations.remote_worker.remote_media import progress_image_from_remote

                    try:
                        percentage = float(preview["percentage"]) if preview.get("percentage") is not None else None
                    except (TypeError, ValueError):
                        percentage = None

                    try:
                        revision = int(preview["revision"]) if preview.get("revision") is not None else None
                    except (TypeError, ValueError):
                        revision = None

                    _emit_progress(
                        services,
                        event_item,
                        invocation,
                        worker,
                        message=str(preview.get("message") or "Rendering"),
                        percentage=percentage,
                        image=progress_image_from_remote(preview),
                        revision=revision,
                    )
                    last_signature = signature
            elif last_signature is None:
                _emit_progress(
                    services,
                    event_item,
                    invocation,
                    worker,
                    message="Rendering",
                )

        time.sleep(settings.poll_interval_seconds)


def _mark_remaining_local_nodes_skipped(context: Any) -> None:
    session = context._data.queue_item.session
    current_exec_id = str(context._data.invocation.id)
    current_source_id = str(session.prepared_source_mapping.get(current_exec_id, current_exec_id))

    for exec_id in list(session.execution_graph.nodes.keys()):
        exec_id = str(exec_id)
        if exec_id == current_exec_id:
            continue
        session.executed.add(exec_id)
        try:
            session._set_prepared_exec_state(exec_id, "skipped")
        except Exception:
            pass
        try:
            session._remove_from_ready_queues(exec_id)
        except Exception:
            pass

    for source_id in list(session.graph.nodes.keys()):
        source_id = str(source_id)
        if source_id == current_source_id:
            continue
        session.executed.add(source_id)
        if source_id not in session.executed_history:
            session.executed_history.append(source_id)


class _RemoteWorkerPool:
    def __init__(self, services: Any, queue_id: str, user_id: str) -> None:
        self.services = services
        self.queue_id = queue_id
        self.user_id = user_id
        self._guard = threading.Lock()
        self._lanes: dict[str, threading.Thread] = {}

    def ensure_workers(self, workers: tuple[WorkerSpec, ...]) -> None:
        with self._guard:
            for worker in workers:
                key = worker.url.casefold()
                thread = self._lanes.get(key)
                if thread is not None and thread.is_alive():
                    continue

                thread = threading.Thread(
                    target=self._run_lane,
                    args=(worker,),
                    daemon=True,
                    name=f"invokeai-remote-pool-{worker.name}",
                )
                self._lanes[key] = thread
                thread.start()

    def _run_lane(self, worker: WorkerSpec) -> None:
        excluded: set[int] = set()
        idle_checks = 0
        last_unavailable_log = 0.0

        while True:
            _park_remote_only_items(self.services, self.queue_id, self.user_id)

            if not _has_eligible_item(
                self.services,
                queue_id=self.queue_id,
                user_id=self.user_id,
                worker=worker,
            ):
                idle_checks += 1
                if idle_checks >= 4:
                    return
                time.sleep(0.25)
                continue

            idle_checks = 0
            lock = _slot_lock(worker)
            if not lock.acquire(blocking=False):
                time.sleep(0.05)
                continue

            try:
                # Do not claim the real local row until this remote is reachable.
                try:
                    from invokeai.app.invocations.remote_worker.remote_nodes import _remote_client

                    _remote_client(worker.url, self.user_id).get_current_item()
                except Exception as exc:
                    now = time.monotonic()
                    if last_unavailable_log == 0.0 or now - last_unavailable_log >= 30.0:
                        self.services.logger.warning(
                            f"Remote Workers [{worker.name}]: unavailable; "
                            f"will retry while eligible work remains: {exc}"
                        )
                        last_unavailable_log = now
                    time.sleep(2.0)
                    continue

                claimed = _claim_for_worker(
                    self.services,
                    queue_id=self.queue_id,
                    user_id=self.user_id,
                    worker=worker,
                    excluded=excluded,
                )
                if claimed is None:
                    time.sleep(0.05)
                    continue

                queue_item, settings = claimed
                try:
                    outcome = _run_remote_job(
                        self.services,
                        queue_item,
                        settings,
                        worker,
                        complete_local=True,
                    )
                except Exception as exc:
                    excluded.add(int(queue_item.item_id))
                    _requeue(
                        self.services,
                        int(queue_item.item_id),
                        settings,
                        f"{worker.name} could not complete it ({exc})",
                    )
                    return

                if outcome == "canceled":
                    continue
            finally:
                lock.release()


def ensure_remote_worker_pool(services: Any, item_id: int) -> bool:
    try:
        item = services.session_queue.get_queue_item(int(item_id))
    except Exception:
        return False

    settings = _settings_for_queue_item(item)
    if settings is None:
        return False

    key = (str(item.queue_id), str(item.user_id))
    with _POOL_LOCK:
        pool = _POOLS.get(key)
        if pool is None:
            pool = _RemoteWorkerPool(services, key[0], key[1])
            _POOLS[key] = pool

    pool.ensure_workers(settings.workers)
    return True


def schedule_remote_worker_pool(item_id: int, services: Any) -> None:
    # Starting lanes is CPU/network-only and must not block the enqueue API.
    threading.Thread(
        target=ensure_remote_worker_pool,
        args=(services, int(item_id)),
        daemon=True,
        name=f"invokeai-remote-pool-start-{item_id}",
    ).start()


def run_current_remote_only(context: Any, invocation: Any) -> None:
    queue_item = context._data.queue_item
    item_id = int(queue_item.item_id)
    settings = _settings_from_helper(invocation)
    if settings is None or settings.mode != "Remote Only":
        raise RemoteInvokeError("Remote Only worker settings are invalid")

    set_status = getattr(context._services.session_queue, "_set_queue_item_status", None)
    if set_status is None:
        raise RemoteInvokeError("Remote Only requires queue status transitions")

    _register_local_handoff(item_id)
    try:
        if _status(context._services, item_id) == "in_progress":
            set_status(
                item_id=item_id,
                status="waiting",
                queue_item=queue_item,
            )
            context.logger.info(f"Remote Workers: Remote Only local item {item_id} is waiting for a free remote worker")

        ensure_remote_worker_pool(context._services, item_id)

        last_error: Exception | None = None
        attempted: set[str] = set()

        while not context.util.is_canceled():
            made_attempt = False

            for worker in settings.workers:
                key = worker.url.casefold()
                if key in attempted:
                    continue

                lock = _slot_lock(worker)
                if not lock.acquire(blocking=False):
                    continue

                made_attempt = True
                attempted.add(key)
                try:
                    if _status(context._services, item_id) == "waiting":
                        set_status(
                            item_id=item_id,
                            status="in_progress",
                            device=f"remote:{worker.name}",
                            queue_item=queue_item,
                        )

                    try:
                        outcome = _run_remote_job(
                            context._services,
                            queue_item,
                            settings,
                            worker,
                            complete_local=False,
                        )
                    except Exception as exc:
                        last_error = exc
                        if not context.util.is_canceled() and _status(context._services, item_id) == "in_progress":
                            set_status(
                                item_id=item_id,
                                status="waiting",
                                queue_item=queue_item,
                            )
                        context.logger.warning(
                            f"Remote Workers [{worker.name}]: could not accept Local-held Remote Only item: {exc}"
                        )
                        continue

                    if outcome == "completed":
                        _mark_remaining_local_nodes_skipped(context)
                        context.util.signal_progress("Remote Only completed", 1.0)
                        return
                    if outcome == "canceled":
                        return
                finally:
                    lock.release()

            if len(attempted) >= len(settings.workers):
                break

            if not made_attempt:
                time.sleep(0.1)

        if context.util.is_canceled():
            return
        if last_error is not None:
            raise RemoteInvokeError(
                f"Remote Only completed with no successful remote worker: {last_error}"
            ) from last_error
        raise RemoteInvokeError("Remote Only completed with no available remote worker")
    finally:
        _unregister_local_handoff(item_id)
