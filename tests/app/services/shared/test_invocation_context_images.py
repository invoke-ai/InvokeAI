from unittest.mock import MagicMock

import pytest

from invokeai.app.services.board_access.board_access_common import (
    BoardAccessDeniedError,
    BoardAction,
    ItemAccessDeniedError,
    ItemAction,
)
from invokeai.app.services.shared.invocation_context import ImagesInterface


def _make_interface(allowed: bool) -> tuple[ImagesInterface, MagicMock]:
    """The board access service decides; these tests check how generation asks it and acts on the answer."""
    services = MagicMock()
    services.configuration.multiuser = True
    services.users.get.return_value = MagicMock(is_admin=False)
    if not allowed:
        services.board_access.require.side_effect = BoardAccessDeniedError("board", BoardAction.AddMedia)
        services.board_access.require_image.side_effect = ItemAccessDeniedError("item", ItemAction.Read)
        services.board_access.require_video.side_effect = ItemAccessDeniedError("item", ItemAction.Read)
    data = MagicMock()
    data.queue_item.user_id = "queue-user"
    data.queue_item.workflow = None
    data.queue_item.session.graph = None
    data.queue_item.session_id = "session"
    data.queue_item.project_id = "project-1"
    data.invocation.is_intermediate = False
    return ImagesInterface(services, data, MagicMock()), services


def test_image_save_rejects_a_board_the_user_cannot_add_to() -> None:
    images, services = _make_interface(allowed=False)

    with pytest.raises(PermissionError, match="not authorized"):
        images.save(MagicMock(), board_id="foreign-board")

    services.images.create.assert_not_called()


def test_image_save_allows_a_board_the_user_can_add_to() -> None:
    images, services = _make_interface(allowed=True)

    images.save(MagicMock(), board_id="public-board")

    services.images.create.assert_called_once()
    services.board_access.require.assert_called_once_with("public-board", "queue-user", False, BoardAction.AddMedia)


def test_image_save_allows_any_board_in_single_user_mode() -> None:
    images, services = _make_interface(allowed=False)
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
    images, services = _make_interface(allowed=False)
    services.image_records.get_user_id.return_value = "owner"
    services.board_image_records.get_board_for_image.return_value = "foreign-board"

    with pytest.raises(PermissionError, match="not authorized"):
        getattr(images, method_name)("private-image")

    getattr(services.images, service_method).assert_not_called()


def test_image_dto_allows_foreign_shared_image() -> None:
    images, services = _make_interface(allowed=True)
    services.image_records.get_user_id.return_value = "owner"
    services.board_image_records.get_board_for_image.return_value = "shared-board"

    images.get_dto("shared-image")

    services.images.get_dto.assert_called_once_with("shared-image")
    services.board_access.require_image.assert_called_once_with("shared-image", "queue-user", False, ItemAction.Read)


def test_image_read_allows_any_image_in_single_user_mode() -> None:
    images, services = _make_interface(allowed=False)
    services.configuration.multiuser = False
    services.users.get.return_value = None

    images.get_dto("foreign-image")

    services.images.get_dto.assert_called_once_with("foreign-image")


def test_image_read_rejects_deactivated_queue_user() -> None:
    """A deactivated account keeps no queue-time privileges, even for its own images."""
    images, services = _make_interface(allowed=False)
    services.users.get.return_value = MagicMock(is_admin=False, is_active=False)
    services.image_records.get_user_id.return_value = "queue-user"

    with pytest.raises(PermissionError, match="not authorized"):
        images.get_dto("own-image")

    services.images.get_dto.assert_not_called()


def test_image_read_rejects_deleted_queue_user() -> None:
    images, services = _make_interface(allowed=False)
    services.users.get.return_value = None

    with pytest.raises(PermissionError, match="not authorized"):
        images.get_dto("any-image")

    services.images.get_dto.assert_not_called()


@pytest.mark.parametrize(
    "account",
    [pytest.param(MagicMock(is_admin=False, is_active=False), id="deactivated"), pytest.param(None, id="deleted")],
)
def test_image_save_rejects_a_deactivated_or_deleted_queue_user(account: MagicMock | None) -> None:
    """No output may be saved on behalf of such an account, even uncategorized."""
    images, services = _make_interface(allowed=False)
    services.users.get.return_value = account

    with pytest.raises(PermissionError, match="not authorized"):
        images.save(MagicMock())

    services.images.create.assert_not_called()


def test_image_save_allows_active_queue_user_without_board() -> None:
    images, services = _make_interface(allowed=False)
    services.users.get.return_value = MagicMock(is_admin=False, is_active=True)

    images.save(MagicMock())

    services.images.create.assert_called_once()


def test_images_save_attributes_the_output_to_the_queue_items_project() -> None:
    images, services = _make_interface(allowed=True)

    images.save(MagicMock())

    assert services.images.create.call_args.kwargs["project_id"] == "project-1"
