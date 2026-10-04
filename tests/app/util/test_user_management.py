"""Coverage for account deletion through the user-management commands."""

import logging
from pathlib import Path
from unittest.mock import Mock

import pytest

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.fonts.fonts_common import FontScope
from invokeai.app.services.fonts.fonts_default import FontService
from invokeai.app.services.image_files.image_files_base import ImageFileStorageBase
from invokeai.app.services.shared.sqlite.sqlite_util import init_db
from invokeai.app.services.users.users_common import UserCreateRequest
from invokeai.app.services.users.users_default import UserService
from invokeai.app.util import user_management


@pytest.fixture
def deletion_fixture(tmp_path: Path):
    config = InvokeAIAppConfig(
        use_memory_db=False,
        db_dir=tmp_path / "databases",
        fonts_dir=tmp_path / "fonts",
        fonts_storage_dir=tmp_path / "uploaded-fonts",
    )
    config._root = tmp_path
    logger = logging.getLogger("user-management-font-tests")
    db = init_db(config, logger, Mock(spec=ImageFileStorageBase))
    fonts = FontService(
        db=db,
        fonts_dir=config.fonts_path,
        storage_dir=config.fonts_storage_path,
        logger=logger,
    )
    users = UserService(db)
    users.create(UserCreateRequest(email="delete-admin@test.com", password="AdminPass123", is_admin=True))
    owner = users.create(UserCreateRequest(email="delete-owner@test.com", password="OwnerPass123"))
    font_bytes = (
        Path(__file__).parents[3] / "invokeai" / "assets" / "fonts" / "inter" / "Inter-Regular.ttf"
    ).read_bytes()
    private = fonts.upload(user_id=owner.user_id, filename="Private.ttf", data=font_bytes).font
    shared = fonts.upload(user_id="system", filename="Shared.ttf", data=font_bytes, scope=FontScope.SHARED).font
    assert private.storage_path is not None
    assert shared.storage_path is not None
    pending = fonts._storage_dir / ("font_" + "a" * 32 + ".ttf")
    pending.write_bytes(font_bytes)
    yield (
        config,
        fonts,
        owner.email,
        shared.id,
        fonts._storage_dir / private.storage_path,
        fonts._storage_dir / shared.storage_path,
        pending,
    )
    fonts.stop()


@pytest.mark.parametrize("interactive", [False, True])
def test_user_delete_commands_clean_private_fonts_and_keep_shared_fonts(
    monkeypatch: pytest.MonkeyPatch, deletion_fixture, interactive: bool
) -> None:
    config, fonts, email, shared_id, private_path, shared_path, pending_path = deletion_fixture
    assert private_path.is_file()
    assert shared_path.is_file()
    assert pending_path.is_file()

    monkeypatch.setattr("invokeai.app.services.config.get_config", lambda: config)
    if interactive:
        answers = iter([email, "yes"])
        monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
        deleted = user_management._delete_user_interactive()
    else:
        deleted = user_management._delete_user_cli(email, force=True)

    assert deleted is True
    assert not private_path.exists()
    assert shared_path.is_file()
    assert pending_path.is_file()
    assert fonts.get_accessible(user_id="system", font_id=shared_id).id == shared_id
