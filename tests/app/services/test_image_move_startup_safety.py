from unittest.mock import MagicMock

from invokeai.app.services.session_processor.session_processor_default import DefaultSessionProcessor


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
