import inspect
import subprocess
import sys
import textwrap
from typing import Any, Callable


def dangerously_run_function_in_subprocess(func: Callable[[], Any]) -> tuple[str, str, int]:
    """**Use with caution! This should _only_ be used with trusted code!**

    Extracts a function's source and runs it in a separate subprocess. Returns stdout, stderr, and return code
    from the subprocess.

    This is useful for tests where an isolated environment is required.

    The function to be called must not have any arguments and must not have any closures over the scope in which is was
    defined.

    Any modules that the function depends on must be imported inside the function.
    """

    source_code = inspect.getsource(func)

    # Must dedent the source code to avoid indentation errors
    dedented_source_code = textwrap.dedent(source_code)

    # Get the function name so we can call it in the subprocess
    func_name = func.__name__

    # Create a script that calls the function
    script = f"""
import sys

{dedented_source_code}

if __name__ == "__main__":
    {func_name}()
"""

    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],  # Run the script in a subprocess
        capture_output=True,  # Capture stdout and stderr
        text=True,
    )

    return result.stdout, result.stderr, result.returncode
