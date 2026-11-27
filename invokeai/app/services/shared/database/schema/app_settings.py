"""Server-wide settings the application keeps in the database."""

from sqlalchemy import Column

from invokeai.app.services.shared.database.schema.metadata import inserted_at, table, updated_at
from invokeai.app.services.shared.database.types import Key, LongText

app_settings = table(
    "app_settings",
    Column("key", Key(), primary_key=True),
    Column("value", LongText(), nullable=False),
    inserted_at(),
    updated_at(),
)
