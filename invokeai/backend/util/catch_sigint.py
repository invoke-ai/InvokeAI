"""
This module defines a context manager `catch_sigint()` which temporarily replaces
the sigINT handler defined by the ASGI in order to allow the user to ^C the application
and shut it down immediately. This was implemented in order to allow the user to interrupt
slow model hashing during startup.

Use like this:

  from invokeai.backend.util.catch_sigint import catch_sigint
  with catch_sigint():
      run_some_hard_to_interrupt_process()
"""

import signal
from contextlib import contextmanager
from typing import Generator


def sigint_handler(signum, frame):  # type: ignore
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.raise_signal(signal.SIGINT)


@contextmanager
def catch_sigint() -> Generator[None, None, None]:
    original_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, sigint_handler)
    yield
    signal.signal(signal.SIGINT, original_handler)
