import datetime
import typing
import uuid

import numpy as np


def get_timestamp() -> int:
    return int(datetime.datetime.now(datetime.timezone.utc).timestamp())


def get_iso_timestamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def get_datetime_from_iso_timestamp(iso_timestamp: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(iso_timestamp)


SEED_MAX = np.iinfo(np.uint32).max


def get_random_seed() -> int:
    rng = np.random.default_rng(seed=None)
    return int(rng.integers(0, SEED_MAX))


def uuid_string() -> str:
    res = uuid.uuid4()
    return str(res)


def is_optional(value: typing.Any) -> bool:
    """Checks if a value is typed as Optional. Note that Optional is sugar for Union[x, None]."""
    return typing.get_origin(value) is typing.Union and type(None) in typing.get_args(value)
