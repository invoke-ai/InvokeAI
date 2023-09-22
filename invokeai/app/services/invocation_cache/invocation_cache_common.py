from __future__ import annotations

from functools import wraps
from threading import Lock
from typing import Callable, ParamSpec, TypeVar

from pydantic import BaseModel, Field

_P = ParamSpec("_P")
_OUT = TypeVar("_OUT")


class InvocationCacheStatus(BaseModel):
    size: int = Field(description="The current size of the invocation cache")
    hits: int = Field(description="The number of cache hits")
    misses: int = Field(description="The number of cache misses")
    enabled: bool = Field(description="Whether the invocation cache is enabled")
    max_size: int = Field(description="The maximum size of the invocation cache")


class ThreadLock:
    """
    This class contains a read lock decorator and a write lock decorator.
    All write operations should be decorated with the `write` decorator, and all
    read operations should be decorated with the `read` decorator.
    When the class is instantiated, it will contain to locks: a read and a write lock.
    Whenever a read operation is running, it will block all write operations but
    not other read operations. When a write operation is running, it will block
    all other read and write opereations.

    Example usage:
        lock = ThreadLock()

        @lock.read
        def a_read_op():
            ...  # only write ops are blocked here

        @lock.write
        def a_write_op():
            ...  # read and write ops are blocked here
    """

    def __init__(self) -> None:
        self._readers = 0  # current reader count
        self._read_lock = Lock()  # locked when reading
        self._write_lock = Lock()  # locked when writing

    def read(self, fn: Callable[_P, _OUT]) -> Callable[_P, _OUT]:
        @wraps(fn)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _OUT:
            try:
                self._readers += 1
                if not self._read_lock.locked():
                    self._read_lock.acquire()
                return fn(*args, **kwargs)
            finally:
                self._readers = -1
                if self._readers == 0:
                    self._read_lock.release()

        return wrapped

    def write(self, fn: Callable[_P, _OUT]) -> Callable[_P, _OUT]:
        @wraps(fn)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _OUT:
            with self._read_lock, self._write_lock:
                return fn(*args, **kwargs)

        return wrapped
