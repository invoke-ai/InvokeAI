"""Add `image_precise_assets`: the precise-pixel asset file an image record owns, if any.

Existing images have none, and stay valid without one. The row is removed with its image (cascade); the asset file
is not, so whoever deletes image files also owns deleting the asset (its file name derives from the image name).
"""

from sqlalchemy import Column, ForeignKey, Table, inspect

from invokeai.app.services.shared.database.schema.metadata import inserted_at
from invokeai.app.services.shared.database.types import BigInt, Key, LongText
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import (
    PortableMigration,
    PortableMigrationContext,
)


def _add_image_precise_assets(context: PortableMigrationContext) -> None:
    # On MySQL and MariaDB the creation commits on its own; a run repeated after a later failure finds it.
    if inspect(context.conn).has_table("image_precise_assets"):
        return
    # The table its foreign key names, as far as the CREATE TABLE needs it.
    Table("images", context.metadata, Column("image_name", Key(), primary_key=True))
    context.create_table(
        "image_precise_assets",
        Column("image_name", Key(), ForeignKey("images.image_name", ondelete="CASCADE"), primary_key=True),
        Column("asset_format", LongText(), nullable=False),
        Column("format_version", BigInt(), nullable=False),
        inserted_at(),
    )


def build_migration() -> PortableMigration:
    return PortableMigration(
        id="2026_11_03_add_image_precise_assets",
        depends_on="2026_10_28_add_moss_image_enhancer_system_prompt",
        callback=_add_image_precise_assets,
    )
