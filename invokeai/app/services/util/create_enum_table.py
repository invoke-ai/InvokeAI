from enum import Enum
import sqlite3
from typing import Type


def create_sql_values_string_from_string_enum(enum: Type[Enum]):
    """
    Creates a string of the form "('value1'), ('value2'), ..., ('valueN')" from a StrEnum.
    """

    delimiter = ", "
    values = [f"('{e.value}')" for e in enum]
    return delimiter.join(values)


def create_enum_table(
    enum: Type[Enum],
    table_name: str,
    primary_key_name: str,
    cursor: sqlite3.Cursor,
):
    """
    Creates and populates a table to be used as a functional enum.
    """

    values_string = create_sql_values_string_from_string_enum(enum)

    cursor.execute(
        f"""--sql
        CREATE TABLE IF NOT EXISTS {table_name} (
            {primary_key_name} TEXT PRIMARY KEY
        );
        """
    )
    cursor.execute(
        f"""--sql
        INSERT OR IGNORE INTO {table_name} ({primary_key_name}) VALUES {values_string};
        """
    )
