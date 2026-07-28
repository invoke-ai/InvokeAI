from unittest.mock import MagicMock

import pytest

from invokeai.app.services.board_records.board_records_common import BoardVisibility
from invokeai.app.services.shared.invocation_context import ImagesInterface


def _make_interface(visibility: BoardVisibility, owner_id: str = "owner") -> tuple[ImagesInterface, MagicMock]:
    services = MagicMock()
    services.configuration.multiuser = True
    services.users.get.return_value = MagicMock(is_admin=False)
    services.boards.get_dto.return_value = MagicMock(user_id=owner_id, board_visibility=visibility)
    data = MagicMock()
    data.queue_item.user_id = "queue-user"
    data.queue_item.workflow = None
    data.queue_item.session.graph = None
    data.queue_item.session_id = "session"
    data.invocation.is_intermediate = False
    return ImagesInterface(services, data, MagicMock()), services


def test_image_save_rejects_foreign_private_board() -> None:
    images, services = _make_interface(BoardVisibility.Private)

    with pytest.raises(PermissionError, match="not authorized"):
        images.save(MagicMock(), board_id="foreign-board")

    services.images.create.assert_not_called()


def test_image_save_allows_foreign_public_board() -> None:
    images, services = _make_interface(BoardVisibility.Public)

    images.save(MagicMock(), board_id="public-board")

    services.images.create.assert_called_once()


def test_image_save_allows_board_owner() -> None:
    images, services = _make_interface(BoardVisibility.Private, owner_id="queue-user")

    images.save(MagicMock(), board_id="owned-board")

    services.images.create.assert_called_once()


def test_image_save_allows_any_board_in_single_user_mode() -> None:
    images, services = _make_interface(BoardVisibility.Private)
    services.configuration.multiuser = False
    services.users.get.return_value = None

    images.save(MagicMock(), board_id="foreign-board")

    services.images.create.assert_called_once()


@pytest.mark.parametrize(
    ("method_name", "service_method"),
    [
        ("get_pil", "get_pil_image"),
        ("get_metadata", "get_metadata"),
        ("get_dto", "get_dto"),
        ("get_path", "get_path"),
    ],
)
def test_image_reads_reject_foreign_private_image(method_name: str, service_method: str) -> None:
    images, services = _make_interface(BoardVisibility.Private)
    services.image_records.get_user_id.return_value = "owner"
    services.board_image_records.get_board_for_image.return_value = "foreign-board"

    with pytest.raises(PermissionError, match="not authorized"):
        getattr(images, method_name)("private-image")

    getattr(services.images, service_method).assert_not_called()


def test_image_dto_allows_foreign_shared_image() -> None:
    images, services = _make_interface(BoardVisibility.Shared)
    services.image_records.get_user_id.return_value = "owner"
    services.board_image_records.get_board_for_image.return_value = "shared-board"

    images.get_dto("shared-image")

    services.images.get_dto.assert_called_once_with("shared-image")


def test_image_read_allows_any_image_in_single_user_mode() -> None:
    images, services = _make_interface(BoardVisibility.Private)
    services.configuration.multiuser = False
    services.users.get.return_value = None

    images.get_dto("foreign-image")

    services.images.get_dto.assert_called_once_with("foreign-image")
