"""Request-size limits shared by more than one router.

These are ceilings on what a single request may ask the server to do, as opposed to what a caller
is allowed to do (`_access.py`) or how a listing is paged (`shared/pagination.py`).
"""

# Copying media is per-item work the server performs synchronously — a record insert and a file
# copy each — so one request's cost is unbounded without a cap. The frontend's largest batch is a
# project board being duplicated, which its own archive limits already keep well under this.
MAX_COPY_BATCH_SIZE = 1000
