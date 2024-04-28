import io
import sys
from typing import Any


class SuppressOutput:
    """Context manager to suppress stdout.

    Example:
    ```
    with SuppressOutput():
        print("This will not be printed")
    ```
    """

    def __enter__(self):
        # Save the original stdout
        self._original_stdout = sys.stdout
        # Redirect stdout to a dummy StringIO object
        sys.stdout = io.StringIO()

    def __exit__(self, *args: Any, **kwargs: Any):
        # Restore stdout
        sys.stdout = self._original_stdout
