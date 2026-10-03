from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from invokeai.app.invocations.remote_worker.remote_client import RemoteInvokeClient, RemoteInvokeError
from invokeai.app.services.board_records.board_records_common import BoardVisibility
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.session_processor.session_processor_common import ProgressImage
from invokeai.app.util.video_thumbnails import probe_video_with_codec


def _model_json(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return value.model_dump_json()
    except Exception:
        try:
            return json.dumps(value, separators=(",", ":"))
        except Exception:
            return None


def progress_image_from_remote(preview: dict[str, Any]) -> ProgressImage | None:
    raw = preview.get("image")
    if not isinstance(raw, dict):
        return None
    data_url = raw.get("dataURL")
    if not isinstance(data_url, str) or not data_url:
        return None
    try:
        return ProgressImage(
            width=int(raw.get("width")),
            height=int(raw.get("height")),
            dataURL=data_url,
        )
    except Exception:
        return None


def _assert_background_save_access(services: Any, user_id: str | None, board_id: str | None) -> None:
    if not getattr(services.configuration, "multiuser", False):
        return
    user = services.users.get(user_id)
    if user is None or not user.is_active:
        raise PermissionError("Queue user is not authorized to save returned remote images")
    if board_id:
        board = services.boards.get_dto(board_id)
        if not user.is_admin and board.user_id != user_id and board.board_visibility != BoardVisibility.Public:
            raise PermissionError("Queue user is not authorized to save returned remote images to this board")


def save_local_image(
    *,
    services: Any,
    queue_item: Any,
    invocation: Any,
    image: Any,
    metadata: str | None,
    board_id: str,
    result_destination: str,
    source_node_id: str | None = None,
) -> Any:
    user_id = getattr(queue_item, "user_id", None)
    target_board_id = (board_id.strip() or None) if result_destination == "gallery" else None
    _assert_background_save_access(services, user_id, target_board_id)

    workflow_json = _model_json(getattr(queue_item, "workflow", None))
    session = getattr(queue_item, "session", None)
    graph_json = _model_json(getattr(session, "graph", None))

    return services.images.create(
        image=image,
        is_intermediate=False,
        image_category=ImageCategory.OTHER if result_destination == "canvas" else ImageCategory.GENERAL,
        board_id=target_board_id,
        metadata=metadata,
        image_origin=ResourceOrigin.INTERNAL,
        workflow=workflow_json,
        graph=graph_json,
        session_id=getattr(queue_item, "session_id", None),
        node_id=source_node_id or getattr(invocation, "id", None),
        user_id=user_id,
    )


def save_local_video(
    *,
    services: Any,
    queue_item: Any,
    invocation: Any,
    video_bytes: bytes,
    metadata: str | None,
    board_id: str,
    result_destination: str,
    source_node_id: str | None = None,
) -> Any:
    """Stage one MP4 alongside outputs/videos; the native service moves it into storage."""
    user_id = getattr(queue_item, "user_id", None)
    target_board_id = (board_id.strip() or None) if result_destination == "gallery" else None
    _assert_background_save_access(services, user_id, target_board_id)

    outputs_path = services.configuration.outputs_path
    if outputs_path is None:
        raise RemoteInvokeError("Primary InvokeAI has no configured outputs path for video import")

    stage_dir = Path(outputs_path) / "videos"
    stage_dir.mkdir(parents=True, exist_ok=True)
    stage_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".irw_remote_", suffix=".mp4", dir=stage_dir, delete=False) as file:
            stage_path = Path(file.name)
            file.write(video_bytes)

        width, height, duration, fps, _codec = probe_video_with_codec(stage_path)
        workflow_json = _model_json(getattr(queue_item, "workflow", None))
        session = getattr(queue_item, "session", None)
        graph_json = _model_json(getattr(session, "graph", None))

        return services.videos.create(
            source_path=stage_path,
            width=width,
            height=height,
            duration=duration,
            fps=fps,
            video_origin=ResourceOrigin.INTERNAL,
            video_category=ImageCategory.OTHER if result_destination == "canvas" else ImageCategory.GENERAL,
            board_id=target_board_id,
            is_intermediate=False,
            metadata=metadata,
            workflow=workflow_json,
            graph=graph_json,
            session_id=getattr(queue_item, "session_id", None),
            node_id=source_node_id or getattr(invocation, "id", None),
            user_id=user_id,
        )
    finally:
        if stage_path is not None:
            stage_path.unlink(missing_ok=True)


def cleanup_remote_media(
    *,
    client: RemoteInvokeClient,
    image_names: list[str],
    video_names: list[str],
    services: Any,
    reason: str,
) -> None:
    """Best-effort cleanup after the primary no longer needs worker media."""
    deleted_images = 0
    deleted_videos = 0

    for remote_name in image_names:
        try:
            client.delete_image(remote_name)
            deleted_images += 1
        except Exception as exc:
            services.logger.warning(f"Remote Workers: {reason}; remote image cleanup failed for {remote_name}: {exc}")

    for remote_name in video_names:
        try:
            client.delete_video(remote_name)
            deleted_videos += 1
        except Exception as exc:
            services.logger.warning(f"Remote Workers: {reason}; remote video cleanup failed for {remote_name}: {exc}")

    services.logger.info(
        f"Remote Workers: remote cleanup deleted {deleted_images}/{len(image_names)} image(s), "
        f"{deleted_videos}/{len(video_names)} video(s)"
    )
