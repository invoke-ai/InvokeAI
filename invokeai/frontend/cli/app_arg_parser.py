from argparse import ArgumentParser, RawTextHelpFormatter

from invokeai.version import __version__

root_help = r"""Sets a root directory for the app. If omitted, the app will search for the root directory in the following order:
- The `$INVOKEAI_ROOT` environment variable
- The currently active virtual environment's parent directory
- `$HOME/invokeai`"""

app_arg_parser = ArgumentParser(description="Invoke Studio", formatter_class=RawTextHelpFormatter)
app_arg_parser.add_argument("--root", type=str, help=root_help)
app_arg_parser.add_argument("--version", action="version", version=__version__, help="Displays the version and exits.")
