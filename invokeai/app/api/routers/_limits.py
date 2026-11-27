"""Request-size limits shared by more than one router.

These are ceilings on what a single request may ask the server to do, as opposed to what a caller
is allowed to do (`_access.py`) or how a listing is paged (`shared/pagination.py`).
"""

# Copying media is per-item work the server performs synchronously — a record insert and a file
# copy each — so one request's cost is unbounded without a cap. The frontend's largest batch is a
# project board being duplicated, which its own archive limits already keep well under this.
MAX_COPY_BATCH_SIZE = 1000

# Every name in a batch body costs at least one DB lookup, so an unbounded list lets an
# authenticated client pin a worker with a single request.
#
# "At least one" is doing real work in that sentence. The authorization helpers in _access.py
# short-circuit for an admin (no query) and a direct owner (one), but a user acting on someone
# else's item on a board they share costs three indexed reads per name: the item's owner, its
# board, and that board's access facts. That is the case the bound has to hold, and it is why no
# route gets a laxer one: the oversized selections this cap rejects are split client-side instead.
MAX_IMAGE_BATCH_SIZE = 1000
MAX_VIDEO_BATCH_SIZE = 1000
