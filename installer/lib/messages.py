# Copyright (c) 2023 Eugene Brodsky (https://github.com/ebr)
"""
Installer user interaction
"""

import os
import platform
from pathlib import Path

from prompt_toolkit import prompt
from prompt_toolkit.completion import PathCompleter
from prompt_toolkit.validation import Validator
from rich import box, print
from rich.console import Console, Group, group
from rich.panel import Panel
from rich.prompt import Confirm
from rich.style import Style
from rich.syntax import Syntax
from rich.text import Text

"""
INVOKE_AI_SRC=https://github.com/invoke-ai/InvokeAI/archive/refs/tags/${INVOKEAI_VERSION}.zip
INSTRUCTIONS=https://invoke-ai.github.io/InvokeAI/installation/INSTALL_AUTOMATED/
TROUBLESHOOTING=https://invoke-ai.github.io/InvokeAI/installation/INSTALL_AUTOMATED/#troubleshooting
"""


OS = platform.uname().system
ARCH = platform.uname().machine

if OS == "Windows":
    # Windows terminals look better without a background colour
    console = Console(style=Style(color="grey74"))
else:
    console = Console(style=Style(color="grey74", bgcolor="grey19"))


def welcome():
    @group()
    def text():
        if (platform_specific := _platform_specific_help()) != "":
            yield platform_specific
            yield ""
        yield Text.from_markup(
            "Some of the installation steps take a long time to run. Please be patient. If the script appears to hang for more than 10 minutes, please interrupt with [i]Control-C[/] and retry.",
            justify="center",
        )

    console.rule()
    print(
        Panel(
            title="[bold wheat1]Welcome to the InvokeAI Installer",
            renderable=text(),
            box=box.DOUBLE,
            expand=True,
            padding=(1, 2),
            style=Style(bgcolor="grey23", color="orange1"),
            subtitle=f"[bold grey39]{OS}-{ARCH}",
        )
    )
    console.line()


def confirm_install(dest: Path) -> bool:
    if dest.exists():
        print(f":exclamation: Directory {dest} already exists :exclamation:")
        dest_confirmed = Confirm.ask(
            ":stop_sign: Are you sure you want to (re)install in this location?",
            default=False,
        )
    else:
        print(f"InvokeAI will be installed in {dest}")
        dest_confirmed = not Confirm.ask("Would you like to pick a different location?", default=False)
    console.line()

    return dest_confirmed


def dest_path(dest=None) -> Path:
    """
    Prompt the user for the destination path and create the path

    :param dest: a filesystem path, defaults to None
    :type dest: str, optional
    :return: absolute path to the created installation directory
    :rtype: Path
    """

    if dest is not None:
        dest = Path(dest).expanduser().resolve()
    else:
        dest = Path.cwd().expanduser().resolve()
    prev_dest = init_path = dest

    dest_confirmed = confirm_install(dest)

    while not dest_confirmed:
        # if the given destination already exists, the starting point for browsing is its parent directory.
        # the user may have made a typo, or otherwise wants to place the root dir next to an existing one.
        # if the destination dir does NOT exist, then the user must have changed their mind about the selection.
        # since we can't read their mind, start browsing at Path.cwd().
        browse_start = (prev_dest.parent if prev_dest.exists() else Path.cwd()).expanduser().resolve()

        path_completer = PathCompleter(
            only_directories=True,
            expanduser=True,
            get_paths=lambda: [browse_start],
            # get_paths=lambda: [".."].extend(list(browse_start.iterdir()))
        )

        console.line()
        console.print(f"[orange3]Please select the destination directory for the installation:[/] \\[{browse_start}]: ")
        selected = prompt(
            ">>> ",
            complete_in_thread=True,
            completer=path_completer,
            default=str(browse_start) + os.sep,
            vi_mode=True,
            complete_while_typing=True
            # Test that this is not needed on Windows
            # complete_style=CompleteStyle.READLINE_LIKE,
        )
        prev_dest = dest
        dest = Path(selected)
        console.line()

        dest_confirmed = confirm_install(dest.expanduser().resolve())

        if not dest_confirmed:
            dest = prev_dest

    dest = dest.expanduser().resolve()

    try:
        dest.mkdir(exist_ok=True, parents=True)
        return dest
    except PermissionError:
        console.print(
            f"Failed to create directory {dest} due to insufficient permissions",
            style=Style(color="red"),
            highlight=True,
        )
    except OSError:
        console.print_exception()

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

    nvidia = (
        "an [gold1 b]NVIDIA[/] GPU (using CUDA™)",
        "cuda",
    )
    nvidia_with_dml = (
        "an [gold1 b]NVIDIA[/] GPU (using CUDA™, and DirectML™ for ONNX) -- ALPHA",
        "cuda_and_dml",
    )
    amd = (
        "an [gold1 b]AMD[/] GPU (using ROCm™)",
        "rocm",
    )
    cpu = (
        "no compatible GPU, or specifically prefer to use the CPU",
        "cpu",
    )
    idk = (
        "I'm not sure what to choose",
        "idk",
    )

    if OS == "Windows":
        options = [nvidia, nvidia_with_dml, cpu]
    if OS == "Linux":
        options = [nvidia, amd, cpu]
    elif OS == "Darwin":
        options = [cpu]
        # future CoreML?

    if len(options) == 1:
        print(f'Your platform [gold1]{OS}-{ARCH}[/] only supports the "{options[0][1]}" driver. Proceeding with that.')
        return options[0][1]

    # "I don't know" is always added the last option
    options.append(idk)

    options = {str(i): opt for i, opt in enumerate(options, 1)}

    console.rule(":space_invader: GPU (Graphics Card) selection :space_invader:")
    console.print(
        Panel(
            Group(
                "\n".join(
                    [
                        f"Detected the [gold1]{OS}-{ARCH}[/] platform",
                        "",
                        "See [deep_sky_blue1]https://invoke-ai.github.io/InvokeAI/#system[/] to ensure your system meets the minimum requirements.",
                        "",
                        "[red3]🠶[/] [b]Your GPU drivers must be correctly installed before using InvokeAI![/] [red3]🠴[/]",
                    ]
                ),
                "",
                "Please select the type of GPU installed in your computer.",
                Panel(
                    "\n".join([f"[dark_goldenrod b i]{i}[/] [dark_red]🢒[/]{opt[0]}" for (i, opt) in options.items()]),
                    box=box.MINIMAL,
                ),
            ),
            box=box.MINIMAL,
            padding=(1, 1),
        )
    )
    choice = prompt(
        "Please make your selection: ",
        validator=Validator.from_callable(
            lambda n: n in options.keys(), error_message="Please select one the above options"
        ),
    )

    if options[choice][1] == "idk":
        console.print(
            "No problem. We will try to install a version that [i]should[/i] be compatible. :crossed_fingers:"
        )

    return options[choice][1]


def simple_banner(message: str) -> None:
    """
    A simple banner with a message, defined here for styling consistency

    :param message: The message to display
    :type message: str
    """

    console.rule(message)


# TODO this does not yet work correctly
def windows_long_paths_registry() -> None:
    """
    Display a message about applying the Windows long paths registry fix
    """

    with open(str(Path(__file__).parent / "WinLongPathsEnabled.reg"), "r", encoding="utf-16le") as code:
        syntax = Syntax(code.read(), line_numbers=True)

    console.print(
        Panel(
            Group(
                "\n".join(
                    [
                        "We will now apply a registry fix to enable long paths on Windows. InvokeAI needs this to function correctly. We are asking your permission to modify the Windows Registry on your behalf.",
                        "",
                        "This is the change that will be applied:",
                        syntax,
                    ]
                )
            ),
            title="Windows Long Paths registry fix",
            box=box.HORIZONTALS,
            padding=(1, 1),
        )
    )


def introduction() -> None:
    """
    Display a banner when starting configuration of the InvokeAI application
    """

    console.rule()

    console.print(
        Panel(
            title=":art: Configuring InvokeAI :art:",
            renderable=Group(
                "",
                "[b]This script will:",
                "",
                "1. Configure the InvokeAI application directory",
                "2. Help download the Stable Diffusion weight files",
                "   and other large models that are needed for text to image generation",
                "3. Create initial configuration files.",
                "",
                "[i]At any point you may interrupt this program and resume later.",
                "",
                "[b]For the best user experience, please enlarge or maximize this window",
            ),
        )
    )
    console.line(2)


def _platform_specific_help() -> str:
    if OS == "Darwin":
        text = Text.from_markup(
            """[b wheat1]macOS Users![/]\n\nPlease be sure you have the [b wheat1]Xcode command-line tools[/] installed before continuing.\nIf not, cancel with [i]Control-C[/] and follow the Xcode install instructions at [deep_sky_blue1]https://www.freecodecamp.org/news/install-xcode-command-line-tools/[/]."""
        )
    elif OS == "Windows":
        text = Text.from_markup(
            """[b wheat1]Windows Users![/]\n\nBefore you start, please do the following:
  1. Double-click on the file [b wheat1]WinLongPathsEnabled.reg[/] in order to
     enable long path support on your system.
  2. Make sure you have the [b wheat1]Visual C++ core libraries[/] installed. If not, install from
     [deep_sky_blue1]https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist?view=msvc-170[/]"""
        )
    else:
        text = ""
    return text
