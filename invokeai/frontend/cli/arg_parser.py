from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from typing import Optional

from invokeai.version import __version__

_root_help = r"""Sets a root directory for the app.
If omitted, the app will search for the root directory in the following order:
- The `$INVOKEAI_ROOT` environment variable
- The currently active virtual environment's parent directory
- `$HOME/invokeai`"""

_parser = ArgumentParser(description="Invoke Studio", formatter_class=RawTextHelpFormatter)
_parser.add_argument("--root", type=str, help=_root_help)
_parser.add_argument("--version", action="version", version=__version__, help="Displays the version and exits.")


class InvokeAIArgs:
    """Helper class for parsing CLI args.

    Args should never be parsed within the application code, only in the CLI entrypoints. Parsing args within the
    application creates conflicts when running tests or when using application modules directly.

    If the args are needed within the application, the consumer should access them from this class.

    Example:
    ```
    # In a CLI wrapper
    from invokeai.frontend.cli.arg_parser import InvokeAIArgs
    InvokeAIArgs.parse_args()

    # In the application
    from invokeai.frontend.cli.arg_parser import InvokeAIArgs
    args = InvokeAIArgs.args
    """

    args: Optional[Namespace] = None

    @staticmethod
    def parse_args() -> Optional[Namespace]:
        """Parse CLI args and store the result."""
        InvokeAIArgs.args = _parser.parse_args()
        return InvokeAIArgs.args
