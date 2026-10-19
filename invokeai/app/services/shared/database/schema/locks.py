"""Named locks: one row per lock, which a transaction locks to serialise work with every other connection."""

from sqlalchemy import Column

from invokeai.app.services.shared.database.schema.metadata import ENUM_LENGTH, table
from invokeai.app.services.shared.database.types import Key

db_locks = table(
    "db_locks",
    # A `DatabaseLock`; migrations add the rows.
    Column("name", Key(ENUM_LENGTH), primary_key=True),
)
