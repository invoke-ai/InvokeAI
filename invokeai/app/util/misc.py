import datetime


def get_timestamp():
    return int(datetime.datetime.now(datetime.timezone.utc).timestamp())
