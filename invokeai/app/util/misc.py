import datetime
import numpy as np


def get_timestamp():
    return int(datetime.datetime.now(datetime.timezone.utc).timestamp())


def get_iso_timestamp() -> str:
    return datetime.datetime.utcnow().isoformat()


def get_datetime_from_iso_timestamp(iso_timestamp: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(iso_timestamp)


SEED_MAX = np.iinfo(np.uint32).max


def get_random_seed():
    rng = np.random.default_rng(seed=None)
    return int(rng.integers(0, SEED_MAX))
