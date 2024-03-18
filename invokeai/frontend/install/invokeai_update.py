"""
Minimalist updater script. Prompts user for the tag or branch to update to and runs
pip install <path_to_git_source>.
"""

import os
import platform
from distutils.version import LooseVersion
from importlib.metadata import PackageNotFoundError, distribution, distributions

import psutil
import requests
from rich import box, print
from rich.console import Console, group
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.style import Style

from invokeai.version import __version__

INVOKE_AI_SRC = "https://github.com/invoke-ai/InvokeAI/archive"
INVOKE_AI_TAG = "https://github.com/invoke-ai/InvokeAI/archive/refs/tags"
INVOKE_AI_BRANCH = "https://github.com/invoke-ai/InvokeAI/archive/refs/heads"
INVOKE_AI_REL = "https://api.github.com/repos/invoke-ai/InvokeAI/releases"

OS = platform.uname().system
ARCH = platform.uname().machine

if OS == "Windows":
    # Windows terminals look better without a background colour
    console = Console(style=Style(color="grey74"))
else:
    console = Console(style=Style(color="grey74", bgcolor="grey19"))


def invokeai_is_running() -> bool:
    for p in psutil.process_iter():
        try:
            cmdline = p.cmdline()
            matches = [x for x in cmdline if x.endswith(("invokeai", "invokeai.exe"))]
            if matches:
                print(
                    f":exclamation: [bold red]An InvokeAI instance appears to be running as process {p.pid}[/red bold]"
                )
                return True
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return False


def get_pypi_versions():
    url = "https://pypi.org/pypi/invokeai/json"
    try:
        data = requests.get(url).json()
    except Exception:
        raise Exception("Unable to fetch version information from PyPi")

    versions = list(data["releases"].keys())
    versions.sort(key=LooseVersion, reverse=True)
    latest_version = [v for v in versions if "rc" not in v][0]
    latest_release_candidate = [v for v in versions if "rc" in v][0]
    return latest_version, latest_release_candidate, versions


def get_torch_extra_index_url() -> str | None:
    """
    Determine torch wheel source URL and optional modules based on the user's OS.
    """

    resolved_url = None

    # In all other cases (like MacOS (MPS) or Linux+CUDA), there is no need to specify the extra index URL.
    torch_package_urls = {
        "windows_cuda": "https://download.pytorch.org/whl/cu121",
        "linux_rocm": "https://download.pytorch.org/whl/rocm5.6",
        "linux_cpu": "https://download.pytorch.org/whl/cpu",
    }

    nvidia_packages_present = (
        len([d.metadata["Name"] for d in distributions() if d.metadata["Name"].startswith("nvidia")]) > 0
    )
    device = "cuda" if nvidia_packages_present else None
    manual_gpu_selection_prompt = (
        "[bold]We tried and failed to guess your GPU capabilities[/] :thinking_face:. Please select the GPU type:"
    )

    if OS == "Linux":
        if not device:
            # do we even need to offer a CPU-only install option?
            print(manual_gpu_selection_prompt)
            print("1: NVIDIA (CUDA)")
            print("2: AMD (ROCm)")
            print("3: No GPU - CPU only")
            answer = Prompt.ask("Choice:", choices=["1", "2", "3"], default="1")
            match answer:
                case "1":
                    device = "cuda"
                case "2":
                    device = "rocm"
                case "3":
                    device = "cpu"

        if device != "cuda":
            resolved_url = torch_package_urls[f"linux_{device}"]

    if OS == "Windows":
        if not device:
            print(manual_gpu_selection_prompt)
            print("1: NVIDIA (CUDA)")
            print("2: No GPU - CPU only")
            answer = Prompt.ask("Your choice:", choices=["1", "2"], default="1")
            match answer:
                case "1":
                    device = "cuda"
                case "2":
                    device = "cpu"

        if device == "cuda":
            resolved_url = torch_package_urls[f"windows_{device}"]

    return resolved_url


def welcome(latest_release: str, latest_prerelease: str):
    @group()
    def text():
        yield f"InvokeAI Version: [bold yellow]{__version__}"
        yield ""
        yield "This script will update InvokeAI to the latest release, or to the development version of your choice."
        yield ""
        yield "[bold yellow]Options:"
        yield f"""[1] Update to the latest [bold]official release[/bold] ([italic]{latest_release}[/italic])
[2] Update to the latest [bold]pre-release[/bold] (may be buggy, database backups are recommended before installation; caveat emptor!) ([italic]{latest_prerelease}[/italic])
[3] Manually enter the [bold]version[/bold] you wish to update to"""

    console.rule()
    print(
        Panel(
            title="[bold wheat1]InvokeAI Updater",
            renderable=text(),
            box=box.DOUBLE,
            expand=True,
            padding=(1, 2),
            style=Style(bgcolor="grey23", color="orange1"),
            subtitle=f"[bold grey39]{OS}-{ARCH}",
        )
    )
    console.line()


def get_extras():
    try:
        distribution("xformers")
        extras = "[xformers]"
    except PackageNotFoundError:
        extras = ""
    return extras


def main():
    if invokeai_is_running():
        print(":exclamation: [bold red]Please terminate all running instances of InvokeAI before updating.[/red bold]")
        input("Press any key to continue...")
        return

    latest_release, latest_prerelease, versions = get_pypi_versions()

    welcome(latest_release, latest_prerelease)

    release = latest_release
    choice = Prompt.ask("Choice:", choices=["1", "2", "3"], default="1")

    if choice == "1":
        release = latest_release
    elif choice == "2":
        release = latest_prerelease
    elif choice == "3":
        while True:
            release = Prompt.ask("Enter an InvokeAI version")
            release.strip()
            if release in versions:
                break
            print(f":exclamation: [bold red]'{release}' is not a recognized InvokeAI release.[/red bold]")

    extras = get_extras()

    console.line()
    force_reinstall = Confirm.ask(
        "[bold]Force reinstallation of all dependencies?[/] This [i]may[/] help fix a broken upgrade, but is usually not necessary.",
        default=False,
    )

    console.line()
    flags = []
    if (index_url := get_torch_extra_index_url()) is not None:
        flags.append(f"--extra-index-url {index_url}")
    if force_reinstall:
        flags.append("--force-reinstall")
    flags = " ".join(flags)

    print(f":crossed_fingers: Upgrading to [yellow]{release}[/yellow]")
    cmd = f'pip install "invokeai{extras}=={release}" --use-pep517 --upgrade {flags}'

    print("")
    print("")
    if os.system(cmd) == 0:
        print(":heavy_check_mark: Upgrade successful")
    else:
        print(":exclamation: [bold red]Upgrade failed[/red bold]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
