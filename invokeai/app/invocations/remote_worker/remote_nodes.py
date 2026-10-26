import json
import time
from typing import Any, Literal

from invokeai.app.invocations.remote_worker.diffusers_transfer import TemporaryDirectoryModelServer
from invokeai.app.invocations.remote_worker.early_dispatch import AUTOMATIC_REMOTE_WORKER_NODE_TYPE
from invokeai.app.invocations.remote_worker.model_transfer import (
    ModelTransferError,
    TemporaryModelServer,
    model_layout_signature,
    resolve_local_model_file,
)
from invokeai.app.invocations.remote_worker.model_transfer_state import (
    another_generation_needs_model,
    register_model_transfer,
    unregister_model_transfer,
)
from invokeai.app.invocations.remote_worker.remote_client import RemoteConfig, RemoteInvokeClient, RemoteInvokeError
from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.invocation_api import (
    BaseInvocation,
    BaseInvocationOutput,
    InputField,
    InvocationContext,
    OutputField,
    invocation,
    invocation_output,
)

# Strip the internal dispatch helper before the graph is sent to a worker.
_HELPER_NODE_TYPES = {AUTOMATIC_REMOTE_WORKER_NODE_TYPE}


def _remote_client(remote_url: str, user_id: str = "") -> RemoteInvokeClient:
    return RemoteInvokeClient(RemoteConfig.from_environment(base_url=remote_url, verify_ssl=False, user_id=user_id))


def _emit_model_transfer_progress(
    *,
    context: InvocationContext,
    invocation: Any,
    remote_index: int,
    model: Any,
    directory: bool,
    phase: str,
    job: dict[str, Any] | None = None,
    error: str = "",
) -> None:
    """Send transfer-only progress to the queue item's owner; never expose LAN URLs or credentials."""
    queue_item = context._data.queue_item
    backend_item_id = int(getattr(queue_item, "item_id", 0) or 0)
    if backend_item_id < 1:
        return
    job = job or {}
    payload = {
        "backend_item_id": backend_item_id,
        "remote_index": remote_index,
        "model_hash": model.hash,
        "name": model.name,
        "directory": directory,
        "phase": phase,
        "bytes": max(0, int(job.get("bytes") or 0)),
        "total_bytes": max(0, int(job.get("total_bytes") or 0)),
    }
    if error:
        payload["error"] = str(error)[:2048]

    try:
        from invokeai.app.services.events.events_common import InvocationProgressEvent

        source_id = queue_item.session.prepared_source_mapping.get(invocation.id, invocation.id)
        context._services.events.dispatch(
            InvocationProgressEvent(
                queue_id=queue_item.queue_id,
                item_id=queue_item.item_id,
                batch_id=queue_item.batch_id,
                origin=queue_item.origin,
                destination=queue_item.destination,
                user_id=queue_item.user_id,
                session_id=queue_item.session_id,
                invocation=invocation.get_event_invocation(),
                invocation_source_id=source_id,
                message="[[IRW_MODEL_TRANSFER]]" + json.dumps(payload, separators=(",", ":")),
                percentage=None,
            )
        )
    except Exception as exc:
        # A failed UI notification must never fail a model transfer.
        context.logger.debug(f"Could not emit remote model transfer progress: {exc}")


class RemoteModelTransferCancelled(CanceledException):
    """The local generation was canceled while preparing a worker model."""


def _remote_model_matches_local(
    remote_client: RemoteInvokeClient,
    model: Any,
    candidate: dict[str, Any],
    expected_layout: tuple[str, str] | None = None,
) -> bool:
    """True only when a same-hash remote record also has a compatible on-disk layout."""
    payload = candidate.get("model") if isinstance(candidate.get("model"), dict) else candidate
    if str(payload.get("hash") or "").strip() != model.hash:
        return False

    # A single model file has no subdirectory layout to validate; the content hash
    # already identifies its bytes.
    if model.path.is_file():
        return True

    key = str(payload.get("key") or "").strip()
    if not key:
        return False
    local_kind, local_signature = expected_layout or model_layout_signature(model.path)
    try:
        remote_layout = remote_client.get_model_layout(key)
    except RemoteInvokeError as exc:
        if "HTTP 404" in str(exc):
            return False
        raise
    return (
        str(remote_layout.get("kind") or "") == local_kind
        and str(remote_layout.get("signature") or "") == local_signature
    )


def _find_compatible_remote_model(remote_client: RemoteInvokeClient, model: Any) -> dict[str, Any] | None:
    """Find a remote copy matching the model hash and, for directories, its on-disk layout."""
    if model.path.is_file():
        return remote_client.get_model_by_hash(model.hash)

    expected_layout = model_layout_signature(model.path)
    for candidate in remote_client.list_models():
        payload = candidate.get("model") if isinstance(candidate.get("model"), dict) else candidate
        if str(payload.get("hash") or "").strip() != model.hash:
            continue
        if _remote_model_matches_local(remote_client, model, candidate, expected_layout):
            return candidate
    return None


def _transfer_missing_model_to_remote(
    *,
    context: InvocationContext,
    invocation: Any,
    remote_client: RemoteInvokeClient,
    remote_index: int,
    identifier: dict[str, Any],
    transfer_host: str,
    timeout_seconds: int,
) -> None:
    """Transfer one missing model, observing native queue cancellation throughout preparation and installation."""
    try:
        model = resolve_local_model_file(context._services, identifier)
    except ModelTransferError as exc:
        raise RemoteInvokeError(str(exc)) from exc

    existing = _find_compatible_remote_model(remote_client, model)
    if existing is not None:
        return

    transfer_id, transfer = register_model_transfer(remote_client.config.base_url, model.hash)
    is_directory = model.path.is_dir()
    job_id: int | None = None
    status = ""
    cancel_sent = False
    cancel_notified = False
    cancel_started: float | None = None
    cancel_error_logged = False
    lock_acquired = False

    def cancelled() -> bool:
        if transfer.cancel_requested.is_set():
            return True
        try:
            item = context._services.session_queue.get_queue_item(int(context._data.queue_item.item_id))
        except Exception:
            return False
        is_cancelled = str(getattr(item.status, "value", item.status)).lower() in {"canceled", "cancelled"}
        if is_cancelled:
            transfer.cancel_requested.set()
        return is_cancelled

    def notify_cancelled() -> None:
        nonlocal cancel_notified
        if cancel_notified:
            return
        cancel_notified = True
        _emit_model_transfer_progress(
            context=context,
            invocation=invocation,
            remote_index=remote_index,
            model=model,
            directory=is_directory,
            phase="cancelled",
        )

    def preparation_should_cancel() -> bool:
        if not cancelled():
            return False
        notify_cancelled()
        return not another_generation_needs_model(transfer)

    def check_cancellation() -> None:
        nonlocal cancel_started, cancel_sent, cancel_error_logged
        if not cancelled():
            return
        notify_cancelled()
        if job_id is None:
            # Before a worker-side install exists, this caller still owns the
            # singleflight preparation. Keep preparing for another live waiter.
            if another_generation_needs_model(transfer):
                return
            raise RemoteModelTransferCancelled("Remote model transfer cancelled")
        if status in {"completed", "error", "cancelled", "canceled", "failed"}:
            raise RemoteModelTransferCancelled("Remote model transfer cancelled")
        # A second live generation may be using the SAME worker-side install.
        # Keep this primary HTTP server alive until that job reaches a terminal state.
        if another_generation_needs_model(transfer):
            return
        if cancel_started is None:
            cancel_started = time.monotonic()
        # Do not interrupt InvokeAI while it moves/registers an already-downloaded model.
        if status in {"running", "installing"}:
            if time.monotonic() - cancel_started > 90:
                raise RemoteModelTransferCancelled("Remote model transfer cancelled during installation")
            return
        if not cancel_sent:
            path = (
                f"/api/v1/remote_workers/diffusers/install/{job_id}"
                if is_directory
                else f"/api/v2/models/install/{job_id}"
            )
            try:
                cancel_install = getattr(remote_client, "cancel_model_install", None)
                if callable(cancel_install):
                    cancel_install(job_id, directory=is_directory)
                else:
                    remote_client._request("DELETE", path)
            except Exception as exc:
                if not cancel_error_logged:
                    context.logger.warning(
                        f"Remote #{remote_index}: model install job {job_id} cancellation request failed: {exc}"
                    )
                    cancel_error_logged = True
            else:
                cancel_sent = True
        # A missing/unresponsive worker must not keep the primary server alive indefinitely.
        if time.monotonic() - cancel_started > 30:
            context.logger.warning(
                f"Remote #{remote_index}: model install job {job_id} cancellation could not be confirmed"
            )
            raise RemoteModelTransferCancelled("Remote model transfer cancellation not confirmed")

    def report_install_progress(job: dict[str, Any]) -> None:
        if cancelled():
            return
        phase = {
            "waiting": "waiting",
            "downloading": "downloading",
            "downloads_done": "downloading",
            "running": "installing",
            "installing": "installing",
            "completed": "verifying",
            "error": "failed",
            "cancelled": "cancelled",
            "canceled": "cancelled",
        }.get(status, "waiting")
        _emit_model_transfer_progress(
            context=context,
            invocation=invocation,
            remote_index=remote_index,
            model=model,
            directory=is_directory,
            phase=phase,
            job=job,
            error=str(job.get("error") or "") if phase == "failed" else "",
        )

    try:
        while not lock_acquired:
            if cancelled():
                notify_cancelled()
                raise RemoteModelTransferCancelled("Remote model transfer cancelled")
            lock_acquired = transfer.shared_lock.acquire(timeout=0.25)
        check_cancellation()
        if _find_compatible_remote_model(remote_client, model) is not None:
            return
        check_cancellation()
        _emit_model_transfer_progress(
            context=context,
            invocation=invocation,
            remote_index=remote_index,
            model=model,
            directory=is_directory,
            phase="preparing",
        )
        server = (
            TemporaryDirectoryModelServer(
                path=model.path,
                remote_url=remote_client.config.base_url,
                advertise_host=transfer_host,
                should_cancel=preparation_should_cancel,
            )
            if is_directory
            else TemporaryModelServer(
                model=model, remote_url=remote_client.config.base_url, advertise_host=transfer_host
            )
        )
        with server:
            check_cancellation()
            size = (
                sum(file.size for file in server.files)
                if isinstance(server, TemporaryDirectoryModelServer)
                else model.path.stat().st_size
            )
            context.logger.warning(
                f"Remote #{remote_index}: required model '{model.name}' is missing; "
                f"serving {model.path.name} ({size / (1024**3):.2f} GiB) directly from this InvokeAI host over the LAN"
            )
            context.logger.info(f"Remote #{remote_index}: temporary model transfer endpoint ready at {server.url}")
            if is_directory:
                assert isinstance(server, TemporaryDirectoryModelServer)
                job = remote_client.install_directory_from_manifest(
                    server.manifest(name=model.name, model_hash=model.hash)
                )
            else:
                job = remote_client.install_model_from_url(server.url, name=model.name)
            try:
                job_id = int(job.get("id"))
            except (TypeError, ValueError) as exc:
                raise RemoteInvokeError(f"Remote model installer returned no usable job id: {job}") from exc
            status = str(job.get("status") or "").lower()
            context.logger.info(
                f"Remote #{remote_index}: InvokeAI model install job {job_id} started for '{model.name}'"
            )

            started = time.monotonic()
            while True:
                check_cancellation()
                try:
                    job = (
                        remote_client.get_directory_install_job(job_id)
                        if is_directory
                        else remote_client.get_model_install_job(job_id)
                    )
                except RemoteInvokeError as exc:
                    if cancelled() and cancel_sent and "HTTP 404" in str(exc):
                        raise RemoteModelTransferCancelled("Remote model transfer cancelled") from exc
                    raise
                status = str(job.get("status") or "").lower()
                report_install_progress(job)
                check_cancellation()
                if status == "completed":
                    break
                if status in {"error", "canceled", "cancelled"}:
                    detail = job.get("error") or job.get("error_type") or "unknown installation error"
                    raise RemoteInvokeError(f"Remote model install job {job_id} ended with status '{status}': {detail}")
                if time.monotonic() - started > float(timeout_seconds):
                    raise RemoteInvokeError(
                        f"Remote model install job {job_id} timed out after {timeout_seconds:g} seconds"
                    )
                time.sleep(1.0)
            context.logger.info(f"Remote #{remote_index}: model install job {job_id} completed with status {status}")

        check_cancellation()
        installed = _find_compatible_remote_model(remote_client, model)
        check_cancellation()
        if installed is None:
            raise RemoteInvokeError(
                f"Remote #{remote_index}: '{model.name}' finished installing but hash {model.hash} "
                "was not found in the remote model manager"
            )
        _emit_model_transfer_progress(
            context=context,
            invocation=invocation,
            remote_index=remote_index,
            model=model,
            directory=is_directory,
            phase="completed",
        )
        remote_payload = installed.get("model") if isinstance(installed.get("model"), dict) else installed
        context.logger.info(
            f"Remote #{remote_index}: verified transferred model '{model.name}' by hash/layout; "
            f"remote key={remote_payload.get('key', 'unknown')}"
        )
    except RemoteModelTransferCancelled:
        notify_cancelled()
        raise
    except ModelTransferError as exc:
        if cancelled():
            notify_cancelled()
            raise RemoteModelTransferCancelled("Remote model transfer cancelled") from exc
        _emit_model_transfer_progress(
            context=context,
            invocation=invocation,
            remote_index=remote_index,
            model=model,
            directory=is_directory,
            phase="failed",
            error=str(exc),
        )
        raise RemoteInvokeError(str(exc)) from exc
    except Exception as exc:
        if cancelled():
            notify_cancelled()
            raise RemoteModelTransferCancelled("Remote model transfer cancelled") from exc
        _emit_model_transfer_progress(
            context=context,
            invocation=invocation,
            remote_index=remote_index,
            model=model,
            directory=is_directory,
            phase="failed",
            error=str(exc),
        )
        raise
    finally:
        if lock_acquired:
            transfer.shared_lock.release()
        unregister_model_transfer(transfer_id)


def _strip_helper_nodes(graph: dict[str, Any]) -> list[str]:
    nodes = graph.get("nodes")
    if not isinstance(nodes, dict):
        raise RemoteInvokeError("Current workflow graph does not contain a nodes object")

    removed = {
        str(node_id)
        for node_id, node in nodes.items()
        if isinstance(node, dict) and str(node.get("type", "")) in _HELPER_NODE_TYPES
    }
    for node_id in removed:
        nodes.pop(node_id, None)

    edges = graph.get("edges")
    if isinstance(edges, list) and removed:
        kept_edges = []
        for edge in edges:
            if not isinstance(edge, dict):
                kept_edges.append(edge)
                continue
            source = edge.get("source") if isinstance(edge.get("source"), dict) else {}
            destination = edge.get("destination") if isinstance(edge.get("destination"), dict) else {}
            if str(source.get("node_id")) in removed or str(destination.get("node_id")) in removed:
                continue
            kept_edges.append(edge)
        graph["edges"] = kept_edges

    if not nodes:
        raise RemoteInvokeError("Nothing remains after removing the internal Remote Worker dispatch helper.")
    return sorted(removed)


def _disable_graph_cache(graph: dict[str, Any]) -> int:
    nodes = graph.get("nodes")
    if not isinstance(nodes, dict):
        return 0
    count = 0
    for node in nodes.values():
        if isinstance(node, dict):
            node["use_cache"] = False
            count += 1
    return count


def _find_media_references(value: Any, found: set[str]) -> None:
    if isinstance(value, list):
        for item in value:
            _find_media_references(item, found)
        return
    if not isinstance(value, dict):
        return
    image_name = value.get("image_name")
    video_name = value.get("video_name")
    if isinstance(image_name, str) and image_name:
        found.add(f"image:{image_name}")
    if isinstance(video_name, str) and video_name:
        found.add(f"video:{video_name}")
    for child in value.values():
        _find_media_references(child, found)


def _graph_media_references(graph: dict[str, Any]) -> list[str]:
    found: set[str] = set()
    nodes = graph.get("nodes")
    if isinstance(nodes, dict):
        for node in nodes.values():
            _find_media_references(node, found)
    return sorted(found)


def _remap_graph_image_names(value: Any, mapped: dict[str, str]) -> int:
    """Rewrite image references (including nested ImageField/list inputs), not other strings."""
    changed = 0
    if isinstance(value, list):
        for entry in value:
            changed += _remap_graph_image_names(entry, mapped)
    elif isinstance(value, dict):
        original = value.get("image_name")
        if isinstance(original, str) and original in mapped:
            value["image_name"] = mapped[original]
            changed += 1
        for entry in value.values():
            changed += _remap_graph_image_names(entry, mapped)
    return changed


def _remap_graph_video_names(value: Any, mapped: dict[str, str]) -> int:
    """Rewrite video references (including nested VideoField/list inputs), not other strings."""
    changed = 0
    if isinstance(value, list):
        for entry in value:
            changed += _remap_graph_video_names(entry, mapped)
    elif isinstance(value, dict):
        original = value.get("video_name")
        if isinstance(original, str) and original in mapped:
            value["video_name"] = mapped[original]
            changed += 1
        for entry in value.values():
            changed += _remap_graph_video_names(entry, mapped)
    return changed


def _transfer_source_videos_to_remote(
    *,
    context: InvocationContext,
    remote_client: RemoteInvokeClient,
    graph: dict[str, Any],
    video_names: list[str],
    remote_index: int,
    uploaded_names: list[str],
) -> None:
    """Copy every distinct local video once per worker before enqueueing the graph."""
    mapped: dict[str, str] = {}
    for local_name in video_names:
        try:
            # InvocationContext performs the authenticated queue owner's read-access check.
            local_path = context.videos.get_path(local_name)
        except Exception as exc:
            raise RemoteInvokeError(
                f"Remote #{remote_index}: cannot read primary source video '{local_name}': {exc}"
            ) from exc
        try:
            mapped[local_name] = remote_client.upload_input_video(local_path)
            uploaded_names.append(mapped[local_name])
        except Exception as exc:
            raise RemoteInvokeError(
                f"Remote #{remote_index}: could not transfer source video '{local_name}': {exc}"
            ) from exc
        context.logger.debug(
            f"Remote #{remote_index}: transferred input video '{local_name}' -> '{mapped[local_name]}'"
        )

    changed = _remap_graph_video_names(graph.get("nodes", {}), mapped)
    context.logger.info(
        f"Remote #{remote_index}: remapped {changed} video field(s) from {len(mapped)} transferred source video(s)"
    )


def _transfer_source_images_to_remote(
    *,
    context: InvocationContext,
    remote_client: RemoteInvokeClient,
    graph: dict[str, Any],
    image_names: list[str],
    remote_index: int,
    uploaded_names: list[str],
) -> None:
    """Copy every distinct local image once per worker before enqueueing the graph."""
    mapped: dict[str, str] = {}
    for local_name in image_names:
        try:
            # InvocationContext checks access for the authenticated queue owner.
            local_image = context.images.get_pil(local_name)
        except Exception as exc:
            raise RemoteInvokeError(
                f"Remote #{remote_index}: cannot read primary source image '{local_name}': {exc}"
            ) from exc
        try:
            mapped[local_name] = remote_client.upload_input_image(local_image)
            uploaded_names.append(mapped[local_name])
        except Exception as exc:
            raise RemoteInvokeError(
                f"Remote #{remote_index}: could not transfer source image '{local_name}': {exc}"
            ) from exc
        context.logger.debug(
            f"Remote #{remote_index}: transferred input image '{local_name}' -> '{mapped[local_name]}'"
        )
    # Never change the local source graph; `graph` is a per-worker deepcopy.
    changed = _remap_graph_image_names(graph.get("nodes", {}), mapped)
    context.logger.info(
        f"Remote #{remote_index}: remapped {changed} image field(s) from {len(mapped)} transferred source image(s)"
    )


def _strip_remote_board_assignments(graph: dict[str, Any]) -> list[str]:
    nodes = graph.get("nodes")
    if not isinstance(nodes, dict):
        return []
    removed_from: list[str] = []
    for node_id, node in nodes.items():
        if not isinstance(node, dict):
            continue
        changed = False
        if isinstance(node.get("board"), dict):
            node["board"] = None
            changed = True
        if "board_id" in node and node.get("board_id") is not None:
            node["board_id"] = None
            changed = True
        if changed:
            removed_from.append(str(node_id))
    return removed_from


@invocation_output("irw_builtin_remote_worker_dispatch_output")
class RemoteWorkerDispatchOutput(BaseInvocationOutput):
    started: bool = OutputField(description="Whether the backend Remote Worker pool was started for this queue item.")


# IMPORTANT: the Python class name intentionally begins with AAA. InvokeAI-7
# groups ready nodes by Python class name and, absent ready_order, selects classes
# alphabetically. This lets the internal Remote Worker helper run before normal
# workflow classes when Local wins the dequeue race.
@invocation(
    AUTOMATIC_REMOTE_WORKER_NODE_TYPE,
    title="Remote Workers - Built-in Worker Pool",
    tags=["remote", "invokeai", "worker", "parallel", "workflow"],
    category="Remote Invoke",
    version="1.0.0",
    use_cache=False,
)
class AAARemoteWorkerDispatchInvocation(BaseInvocation):
    """Internal helper that connects a normal InvokeAI queue item to the backend Remote Worker pool."""

    result_destination: Literal["gallery", "canvas"] = InputField(
        default="gallery",
        ui_hidden=True,
        description="Internal: captured InvokeAI result destination (Gallery or Canvas).",
    )
    local_gallery_board_id: str = InputField(
        default="",
        ui_hidden=True,
        description="Internal: Gallery board captured when this workflow was queued.",
    )
    remote_url: str = InputField(
        default="",
        ui_hidden=True,
        description="Internal: primary remote InvokeAI URL.",
    )
    additional_remote_urls: str = InputField(
        default="",
        ui_hidden=True,
        description="Internal: additional remote InvokeAI URLs.",
    )
    dispatch_mode: Literal["Distributed", "Remote Only"] = InputField(
        default="Distributed",
        ui_hidden=True,
        description="Internal automatic worker-pool mode.",
    )
    remote_worker_names: str = InputField(
        default="[]",
        ui_hidden=True,
        description="Internal JSON list of user-defined worker names aligned with the configured URLs.",
    )
    keep_remote_copies: bool = InputField(
        default=False,
        ui_hidden=True,
        description="Keep generated media on remote workers after successful import.",
    )
    auto_transfer_missing_models: bool = InputField(
        default=True,
        ui_hidden=True,
        description="Transfer supported missing models to a worker before rendering.",
    )
    model_transfer_host: str = InputField(
        default="",
        ui_hidden=True,
        description="Optional primary LAN host used by workers during model transfer.",
    )
    model_transfer_timeout_seconds: int = InputField(
        default=7200,
        ge=60,
        le=86400,
        ui_hidden=True,
        description="Maximum time to wait for a remote model transfer/install job.",
    )
    collector_poll_interval_seconds: float = InputField(
        default=0.75,
        ge=0.25,
        le=30.0,
        ui_hidden=True,
        description="How often worker-pool lanes poll remote progress and status.",
    )
    collector_timeout_seconds: int = InputField(
        default=14400,
        ge=10,
        le=86400,
        ui_hidden=True,
        description="Maximum wait per remote queued/rendering phase.",
    )

    def invoke(self, context: InvocationContext) -> RemoteWorkerDispatchOutput:
        from invokeai.app.invocations.remote_worker.worker_pool import (
            ensure_remote_worker_pool,
            run_current_remote_only,
        )

        item_id = int(context._data.queue_item.item_id)
        if self.dispatch_mode == "Remote Only":
            run_current_remote_only(context, self)
        else:
            ensure_remote_worker_pool(context._services, item_id)

        return RemoteWorkerDispatchOutput(started=True)
