"""The precise-pixel asset an image record owns, beside the image file itself."""

from sqlalchemy import Column, ForeignKey

from invokeai.app.services.shared.database.schema.metadata import inserted_at, table
from invokeai.app.services.shared.database.types import BigInt, Key, LongText

# An image has at most one asset, keyed by the image name. The asset file's name derives from the image name, so
# the row holds only what the file cannot tell without being read. Internal bookkeeping, never part of an image DTO.
image_precise_assets = table(
    "image_precise_assets",
    Column("image_name", Key(), ForeignKey("images.image_name", ondelete="CASCADE"), primary_key=True),
    # The file format, currently only 'raster_safetensors'.
    Column("asset_format", LongText(), nullable=False),
    Column("format_version", BigInt(), nullable=False),
    inserted_at(),
)
