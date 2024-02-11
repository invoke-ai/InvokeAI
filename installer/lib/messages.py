# Copyright (c) 2023 Eugene Brodsky (https://github.com/ebr)
"""
Installer user interaction
"""

import os
import platform
from enum import Enum
from pathlib import Path

from prompt_toolkit import HTML, prompt
from prompt_toolkit.completion import FuzzyWordCompleter, PathCompleter
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


def welcome(available_releases: tuple | None = None) -> None:
    @group()
    def text():
        if (platform_specific := _platform_specific_help()) is not None:
            yield platform_specific
            yield ""
        yield Text.from_markup(
            "Some of the installation steps take a long time to run. Please be patient. If the script appears to hang for more than 10 minutes, please interrupt with [i]Control-C[/] and retry.",
            justify="center",
        )
        if available_releases is not None:
            latest_stable = available_releases[0][0]
            last_pre = available_releases[1][0]
            yield ""
            yield Text.from_markup(
                f"[red3]🠶[/] Latest stable release (recommended): [b bright_white]{latest_stable}", justify="center"
            )
            yield Text.from_markup(
                f"[red3]🠶[/] Last published pre-release version: [b bright_white]{last_pre}", justify="center"
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


def choose_version(available_releases: tuple | None = None) -> str:
    """
    Prompt the user to choose an Invoke version to install
    """

    # short circuit if we couldn't get a version list
    # still try to install the latest stable version
    if available_releases is None:
        return "stable"

    console.print(f"   Version {choices[0] if response == '' else response} will be installed.")

    console.line()

    return "stable" if response == "" else response


def user_wants_auto_configuration() -> bool:
    """Prompt the user to choose between manual and auto configuration."""
    console.rule("InvokeAI Configuration Section")
    console.print(
        Panel(
            Group(
                "\n".join(
                    [
                        "Libraries are installed and InvokeAI will now set up its root directory and configuration. Choose between:",
                        "",
                        "  * AUTOMATIC configuration:  install reasonable defaults and a minimal set of starter models.",
                        "  * MANUAL configuration: manually inspect and adjust configuration options and pick from a larger set of starter models.",
                        "",
                        "Later you can fine tune your configuration by selecting option [6] 'Change InvokeAI startup options' from the invoke.bat/invoke.sh launcher script.",
                    ]
                ),
            ),
            box=box.MINIMAL,
            padding=(1, 1),
        )
    )
    choice = (
        prompt(
            HTML("Choose <b>&lt;a&gt;</b>utomatic or <b>&lt;m&gt;</b>anual configuration [a/m] (a): "),
            validator=Validator.from_callable(
                lambda n: n == "" or n.startswith(("a", "A", "m", "M")), error_message="Please select 'a' or 'm'"
            ),
        )
        or "a"
    )
    return choice.lower().startswith("a")


def confirm_install(dest: Path) -> bool:
    if dest.exists():
        print(f":stop_sign: Directory {dest} already exists!")
        print("   Is this location correct?")
        default = False
    else:
        print(f":file_folder: InvokeAI will be installed in {dest}")
        default = True

    dest_confirmed = Confirm.ask("   Please confirm:", default=default)

    console.line()

    return dest_confirmed


def dest_path(dest=None) -> Path | None:
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
    dest_confirmed = False

    while not dest_confirmed:
        browse_start = (dest or Path.cwd()).expanduser().resolve()

        path_completer = PathCompleter(
            only_directories=True,
            expanduser=True,
            get_paths=lambda: [str(browse_start)],  # noqa: B023
            # get_paths=lambda: [".."].extend(list(browse_start.iterdir()))
        )

        console.line()

        console.print(f":grey_question: [orange3]Please select the install destination:[/] \\[{browse_start}]: ")
        selected = prompt(
            ">>> ",
            complete_in_thread=True,
            completer=path_completer,
            default=str(browse_start) + os.sep,
            vi_mode=True,
            complete_while_typing=True,
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


class GpuType(Enum):
    CUDA = "cuda"
    CUDA_AND_DML = "cuda_and_dml"
    ROCM = "rocm"
    CPU = "cpu"
    AUTODETECT = "autodetect"


def select_gpu() -> GpuType:
    """
    Prompt the user to select the GPU driver
    """

    if ARCH == "arm64" and OS != "Darwin":
        print(f"Only CPU acceleration is available on {ARCH} architecture. Proceeding with that.")
        return GpuType.CPU

    nvidia = (
        "an [gold1 b]NVIDIA[/] GPU (using CUDA™)",
        GpuType.CUDA,
    )
    nvidia_with_dml = (
        "an [gold1 b]NVIDIA[/] GPU (using CUDA™, and DirectML™ for ONNX) -- ALPHA",
        GpuType.CUDA_AND_DML,
    )
    amd = (
        "an [gold1 b]AMD[/] GPU (using ROCm™)",
        GpuType.ROCM,
    )
    cpu = (
        "Do not install any GPU support, use CPU for generation (slow)",
        GpuType.CPU,
    )
    autodetect = (
        "I'm not sure what to choose",
        GpuType.AUTODETECT,
    )

    options = []
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
    options.append(autodetect)  # type: ignore

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

    if options[choice][1] is GpuType.AUTODETECT:
        console.print(
            "No problem. We will install CUDA support first :crossed_fingers: If Invoke does not detect a GPU, please re-run the installer and select one of the other GPU types."
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
        syntax = Syntax(code.read(), line_numbers=True, lexer="regedit")

    console.print(
        Panel(
            Group(
                "\n".join(
                    [
                        "We will now apply a registry fix to enable long paths on Windows. InvokeAI needs this to function correctly. We are asking your permission to modify the Windows Registry on your behalf.",
                        "",
                        "This is the change that will be applied:",
                        str(syntax),
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


def _platform_specific_help() -> Text | None:
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
        return
    return text
