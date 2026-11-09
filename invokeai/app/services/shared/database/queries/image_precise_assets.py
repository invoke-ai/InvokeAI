"""The precise-pixel asset each image record may own: the file's format and version.

Internal bookkeeping for image-file storage; no image DTO carries it. The asset's file name derives from the image
name. Deleting an image record removes its row (cascade) but not the file.
"""

import itertools
from collections.abc import Sequence
from typing import Any, NamedTuple, Optional

from sqlalchemy import Connection, Row, bindparam, insert, select

from invokeai.app.services.shared.database.queries.base import IN_CHUNK, QueryModule, mapped, read, write
from invokeai.app.services.shared.database.schema.image_precise_assets import image_precise_assets

_A = image_precise_assets.c


class PreciseAssetRecord(NamedTuple):
    image_name: str
    asset_format: str
    format_version: int


_GET = select(_A.image_name, _A.asset_format, _A.format_version).where(_A.image_name == bindparam("image_name"))
_NAMES_WITH_ASSETS = select(_A.image_name).where(_A.image_name.in_(bindparam("names", expanding=True)))
_INSERT = insert(image_precise_assets)


def _record(row: Optional[Row[Any]]) -> Optional[PreciseAssetRecord]:
    return None if row is None else PreciseAssetRecord(*row)


class ImagePreciseAssetQueries(QueryModule):
    @mapped(_record)
    @read
    def get(self, conn: Connection, image_name: str) -> Optional[Row[Any]]:
        return conn.execute(_GET, {"image_name": image_name}).first()

    @read
    def names_with_assets(self, conn: Connection, image_names: Sequence[str]) -> set[str]:
        """Which of these images have an asset."""
        names: set[str] = set()
        for chunk in itertools.batched(image_names, IN_CHUNK):
            names.update(conn.execute(_NAMES_WITH_ASSETS, {"names": list(chunk)}).scalars().all())
        return names

    @write
    def insert(self, conn: Connection, *, image_name: str, asset_format: str, format_version: int) -> None:
        """Raises `UniqueViolation` when the image already has an asset and `ForeignKeyViolation` when there is no
        such image."""
        conn.execute(
            _INSERT, {"image_name": image_name, "asset_format": asset_format, "format_version": format_version}
        )
