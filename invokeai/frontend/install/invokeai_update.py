"""
Minimalist updater script. Prompts user for the tag or branch to update to and runs
pip install <path_to_git_source>.
"""
import os
import platform
from distutils.version import LooseVersion

import pkg_resources
import psutil
import requests
from rich import box, print
from rich.console import Console, group
from rich.panel import Panel
from rich.prompt import Prompt
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
    extras = ""
    try:
        _ = pkg_resources.get_distribution("xformers")
        extras = "[xformers]"
    except pkg_resources.DistributionNotFound:
        pass
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

    print(f":crossed_fingers: Upgrading to [yellow]{release}[/yellow]")
    cmd = f'pip install "invokeai{extras}=={release}" --use-pep517 --upgrade'

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
