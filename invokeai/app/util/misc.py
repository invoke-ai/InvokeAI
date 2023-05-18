import datetime
import numpy as np


def get_timestamp():
    return int(datetime.datetime.now(datetime.timezone.utc).timestamp())


SEED_MAX = np.iinfo(np.int32).max


def get_random_seed():
    return np.random.randint(0, SEED_MAX)
