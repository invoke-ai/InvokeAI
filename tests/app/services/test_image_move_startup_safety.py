from unittest.mock import MagicMock

from invokeai.app.services.invocation_services import InvocationServices
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_processor.session_processor_default import DefaultSessionProcessor


def _services(**overrides):
    services = {
        "board_image_records": object(),
        "board_images": object(),
        "board_records": object(),
        "boards": object(),
        "bulk_download": object(),
        "configuration": object(),
        "events": object(),
        "images": object(),
        "image_files": object(),
        "image_records": object(),
        "logger": object(),
        "model_images": object(),
        "model_manager": object(),
        "model_relationships": object(),
        "model_relationship_records": object(),
        "download_queue": object(),
        "external_generation": object(),
        "performance_statistics": object(),
        "session_queue": object(),
        "session_processor": object(),
        "invocation_cache": object(),
        "names": object(),
        "urls": object(),
        "workflow_records": object(),
        "tensors": object(),
        "conditioning": object(),
        "style_preset_records": object(),
        "style_preset_image_files": object(),
        "workflow_thumbnails": object(),
        "client_state_persistence": object(),
        "users": object(),
        "image_moves": None,
    }
    services.update(overrides)
    return InvocationServices(**services)


def test_image_moves_start_before_session_processor() -> None:
    started: list[str] = []
    image_moves = MagicMock()
    image_moves.start.side_effect = lambda _invoker: started.append("image_moves")
    session_processor = MagicMock()
    session_processor.start.side_effect = lambda _invoker: started.append("session_processor")

    Invoker(_services(image_moves=image_moves, session_processor=session_processor))

    assert started == ["image_moves", "session_processor"]


def test_session_processor_detects_active_image_move_maintenance() -> None:
    image_moves = MagicMock()
    image_moves.is_maintenance_active.return_value = True
    processor = DefaultSessionProcessor()
    processor._invoker = MagicMock()
    processor._invoker.services.image_moves = image_moves

    assert processor._is_image_move_maintenance_active() is True


def test_session_processor_allows_processing_without_image_move_maintenance() -> None:
    image_moves = MagicMock()
    image_moves.is_maintenance_active.return_value = False
    processor = DefaultSessionProcessor()
    processor._invoker = MagicMock()
    processor._invoker.services.image_moves = image_moves

    assert processor._is_image_move_maintenance_active() is False
