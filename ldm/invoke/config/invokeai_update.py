'''
Minimalist updater script. Prompts user for the tag or branch to update to and runs
pip install <path_to_git_source>.
'''

import platform
import requests
import subprocess
from rich import box, print
from rich.console import Console, group
from rich.panel import Panel
from rich.prompt import Prompt
from rich.style import Style
from rich.text import Text
from rich.live import Live
from rich.table import Table

from ldm.invoke import __version__

INVOKE_AI_SRC="https://github.com/invoke-ai/InvokeAI/archive"
INVOKE_AI_REL="https://api.github.com/repos/invoke-ai/InvokeAI/releases"

OS = platform.uname().system
ARCH = platform.uname().machine

ORANGE_ON_DARK_GREY = Style(bgcolor="grey23", color="orange1")

if OS == "Windows":
    # Windows terminals look better without a background colour
    console = Console(style=Style(color="grey74"))
else:
    console = Console(style=Style(color="grey74", bgcolor="grey23"))

def get_versions()->dict:
    return requests.get(url=INVOKE_AI_REL).json()

def welcome(versions: dict):

    @group()
    def text():
        yield f'InvokeAI Version: [bold yellow]{__version__}'
        yield ''
        yield 'This script will update InvokeAI to the latest release, or to a development version of your choice.'
        yield ''
        yield '[bold yellow]Options:'
        yield f'''[1] Update to the latest official release ([italic]{versions[0]['tag_name']}[/italic])
[2] Update to the bleeding-edge development version ([italic]main[/italic])
[3] Manually enter the tag or branch name you wish to update'''

    console.rule()
    console.print(
        Panel(
            title="[bold wheat1]InvokeAI Updater",
            renderable=text(),
            box=box.DOUBLE,
            expand=True,
            padding=(1, 2),
            style=ORANGE_ON_DARK_GREY,
            subtitle=f"[bold grey39]{OS}-{ARCH}",
        )
    )
    # console.rule is used instead of console.line to maintain dark background
    # on terminals where light background is the default
    console.rule(characters=" ")

def main():
    versions = get_versions()
    welcome(versions)

    tag = None
    choice = Prompt.ask(Text.from_markup(('[grey74 on grey23]Choice:')),choices=['1','2','3'],default='1')

    if choice=='1':
        tag = versions[0]['tag_name']
    elif choice=='2':
        tag = 'main'
    elif choice=='3':
        tag = Prompt.ask('[grey74 on grey23]Enter an InvokeAI tag or branch name')

    console.print(Panel(f':crossed_fingers: Upgrading to [yellow]{tag}[/yellow]', box=box.MINIMAL, style=ORANGE_ON_DARK_GREY))

    cmd = f'pip install {INVOKE_AI_SRC}/{tag}.zip --use-pep517'

    progress = Table.grid(expand=True)
    progress_panel = Panel(progress, box=box.MINIMAL, style=ORANGE_ON_DARK_GREY)

    with subprocess.Popen(['bash', '-c', cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        progress.add_column()
        with Live(progress_panel, console=console, vertical_overflow='visible'):
            while proc.poll() is None:
                for l in iter(proc.stdout.readline, b''):
                    progress.add_row(l.decode().strip(), style=ORANGE_ON_DARK_GREY)
        if proc.returncode == 0:
            console.rule(f':heavy_check_mark: Upgrade successful')
        else:
            console.rule(f':exclamation: [bold red]Upgrade failed[/red bold]')

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
