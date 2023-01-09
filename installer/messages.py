"""
Installer user interaction
"""

import platform
from pathlib import Path
from tkinter.filedialog import askdirectory

from rich import box, print
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.style import Style
from rich.text import Text

console = Console(width=80)

OS = platform.uname().system
ARCH = platform.uname().machine


def welcome():
    console.rule()
    print(
        Panel(
            title="[bold wheat1]Welcome to the InvokeAI Installer",
            renderable=Text(
                "Some of the installation steps take a long time to run. Please be patient. If the script appears to hang for more than 10 minutes, please interrupt with control-C and retry.",
                justify="center",
            ),
            box=box.DOUBLE,
            width=80,
            expand=False,
            padding=(1, 2),
            style=Style(bgcolor="grey23", color="orange1"),
            subtitle=f"[bold grey39]{OS}-{ARCH}",
        )
    )
    console.line()


def dest_path(init_path=None) -> Path:
    """
    Prompt the user for the destination path and create the path

    :param init_path: a filesystem path, defaults to None
    :type init_path: str, optional
    :return: absolute path to the created installation directory
    :rtype: Path
    """

    dest = init_path
    dest_confirmed = False

    while not dest_confirmed:
        console.line()
        if dest is not None:
            dest = Path(dest).expanduser().resolve()
            print(f"InvokeAI will be installed at {dest}")
            dest_confirmed = Confirm.ask(f"Continue?")
        if not dest_confirmed:
            print(f"Please select the destination directory for the installation")
            resp = askdirectory(initialdir=dest)
            if resp == ():
                continue
            dest = Path(resp).expanduser().resolve()
        if dest.exists():
            print(f":exclamation: Directory {dest} already exists.")
            dest_confirmed = Confirm.ask(
                ":question: Are you sure you want to (re)install in this location?", default="y"
            )

    try:
        dest.mkdir(exist_ok=True, parents=True)
        return dest
    except PermissionError as exc:
        print(
            f"Failed to create directory {dest} due to insufficient permissions",
            style=Style(color="red"),
            highlight=True,
        )
    except OSError as exc:
        console.print_exception(exc)

    if Confirm.ask("Would you like to try again?"):
        dest_path(init_path)
    else:
        console.rule("Goodbye!")
