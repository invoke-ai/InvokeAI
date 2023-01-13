"""
Installer user interaction
"""

import platform
from pathlib import Path

from prompt_toolkit import prompt
from prompt_toolkit.completion import PathCompleter
from prompt_toolkit.shortcuts import CompleteStyle
from prompt_toolkit.validation import Validator
from rich import box, print
from rich.console import Console, Group
from rich.panel import Panel
from rich.prompt import Confirm
from rich.style import Style
from rich.text import Text

console = Console(style=Style(color="grey74", bgcolor="grey19"))

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

    # TODO: feels like this could be refactored for clarity (@ebr)

    dest = init_path
    if dest is not None:
        dest = Path(dest).expanduser().resolve()

    dest_confirmed = False

    while not dest_confirmed:
        console.line()

        print(f"InvokeAI will be installed at {dest}")
        dest_confirmed = Confirm.ask(f"Is this correct?", default="y")

        if not dest_confirmed:

            # needs more thought into how to handle this nicely
            # so that the first selected destination continues to shows up as
            # default until the user is done selecting (potentially multiple times)
            old_dest = dest

            path_completer = PathCompleter(
                only_directories=True,
                expanduser=True,
                get_paths=lambda: [Path(dest).parent],
                file_filter=lambda n: not n.startswith("."),
            )
            print(f"Please select the destination directory for the installation \[{dest}]: ")
            selected = prompt(
                "[Tab] to complete ⮞ ",
                complete_in_thread=True,
                completer=path_completer,
                complete_style=CompleteStyle.READLINE_LIKE,
            )
            print(selected)
            if Path(selected).is_absolute():
                # use the absolute path directly
                dest = Path(selected)
            else:
                # the user entered a relative path - offer to create it as a sibling to the original destination
                dest = dest.parent / Path(selected)
            dest = dest.expanduser().resolve()

        if dest.exists():
            console.line()
            print(f":exclamation: Directory {dest} already exists.")
            console.line()
            dest_confirmed = Confirm.ask(
                ":question: Are you sure you want to (re)install in this location?", default="y"
            )
            if not dest_confirmed:
                dest = old_dest

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


def graphical_accelerator():
    """
    Prompt the user to select the graphical accelerator in their system
    This does not validate user's choices (yet), but only offers choices
    valid for the platform.
    CUDA is the fallback.
    We may be able to detect the GPU driver by shelling out to `modprobe` or `lspci`,
    but this is not yet supported or reliable. Also, some users may have exotic preferences.
    """


    if ARCH == "arm64" and OS != "Darwin":
        print(f"Only CPU acceleration is available on {ARCH} architecture. Proceeding with that.")
        return "cpu"

    nvidia = ("an [gold1 b]NVIDIA[/] GPU (using CUDA™)", "cuda",)
    amd = ("an [gold1 b]AMD[/] GPU (using ROCm™)", "rocm",)
    cpu = ("no compatible GPU, or specifically prefer to use the CPU", "cpu",)
    idk = ("I'm not sure what to choose", "idk",)

    if OS == "Windows":
        options = [nvidia, cpu]
    if OS == "Linux":
        options = [nvidia, amd, cpu]
    elif OS == "Darwin":
        options = [cpu]
        # future CoreML?

    if len(options) == 1:
        print(f"Your operating system only supports the \"{options[0][1]}\" driver. Proceeding with that.")
        return options[0][1]

    # "I don't know" is always added the last option
    options.append(idk)

    options = {str(i): opt for i, opt in enumerate(options, 1)}

    console.rule(":space_invader: GPU (Graphics Card) selection :space_invader:")
    console.print(Panel(
            Group("\n".join([
                f"Detected the [gold1]{OS}-{ARCH}[/] platform",
                "",
                "See [steel_blue3]https://invoke-ai.github.io/InvokeAI/#system[/] to ensure your system meets the minimum requirements.",
                "",
                "[red3]🠶[/] [b]Your GPU drivers must be correctly installed before using InvokeAI![/] [red3]🠴[/]"]),
                "",
                "Please select the type of GPU installed in your computer.",
                Panel("\n".join([f"[spring_green3 b i]{i}[/] [dark_red]🢒[/]{opt[0]}" for (i, opt) in options.items()]), box=box.MINIMAL),
            ),
            box=box.MINIMAL,
            padding=(1, 1),
        )
    )
    choice = prompt("Please make your selection: ", validator=Validator.from_callable(lambda n: n in options.keys(), error_message="Please select one the above options"))

    if options[choice][1] == "idk":
        console.print("No problem. We will try to install a version that [i]should[/i] be compatible. :crossed_fingers:")

    return options[choice][1]


def simple_banner(message: str) -> None:
    """
    A simple banner with a message, defined here for styling consistency

    :param message: The message to display
    :type message: str
    """

    console.rule(message)


def introduction() -> None:
    """
    Display a banner when starting configuration of the InvokeAI application
    """

    console.rule()

    console.print(Panel(title=":art: Configuring InvokeAI :art:", renderable=Group(
        "",
        "[b]This script will:",
        "",
        "1. Configure the InvokeAI application directory",
        "2. Help download the Stable Diffusion weight files",
        "   and other large models that are needed for text to image generation",
        "3. Create initial configuration files.",
        "",
        "[i]At any point you may interrupt this program and resume later.",
    )))
    console.line(2)