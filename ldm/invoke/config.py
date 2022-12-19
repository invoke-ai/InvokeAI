# from omegaconf import OmegaConf
from pathlib import Path
import os
import shutil

# import readline

from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from ldm.invoke import readline

console = Console()


class RuntimeDir:
    """
    Manages the runtime directory
    """

    def __init__(self, paths) -> None:
        self.paths = paths

    def root(self) -> Path:

        if self.paths.root_arg_passed:
            console.print(f"User specified InvokeAI root path")
        else:
            for ev in ["INVOKEAI_ROOT", "VIRTUAL_ENV"]:
                if os.getenv(ev) is not None:
                    console.print(f"{ev}={os.getenv(ev)} found in environment")

        console.print(
            f"Using {self.paths.root.location.expanduser().resolve()} as the InvokeAI runtime directory"
        )
        console.line()

    def validate(self) -> bool:
        """
        Validate that the runtime dir is correctly configured
        """

        console.rule(
            f"Validating directory structure at {self.paths.root.location.expanduser().resolve()}"
        )
        console.line()

        missing = False
        for path in self.paths.get():
            abspath = path.location.expanduser().resolve()
            if abspath.exists():
                msg_prefix = "[bold bright_green]🗸[/]"
            else:
                msg_prefix = "[bold bright_red]✗[/]"
                missing = True
            console.print(f"{msg_prefix} {path.description} {path.kind}: {abspath}")

        console.line()
        return not missing

    def select_outdir(self):
        current = str(self.paths.outdir.location.expanduser().resolve())

        # readline.parse_and_bind("tab: complete")
        directory = Prompt.ask(
            prompt="Select the default directory for image outputs",
            default=current,
        )
        # readline.parse_and_bind("/")

        self.paths.outdir = directory
        return directory

    def copy_configdir(self):
        """
        Copy source config dir
        this is only needed for source install. Auto install would have already created it

        TODO this is brittle! will break if we move source config dir
        assumes the "configs" dir exists 2 levels up from this file.
        there needs to be a better way of determining this
        """

        src = Path(__file__).parents[2] / "configs"
        dest = self.paths.configdir.location
        shutil.copytree(src, dest, dirs_exist_ok=True)

    @staticmethod
    def safety_checker_config(yes_to_all=False):

        # can only be disabled interactively
        # TODO also accept this as a flag in config script
        enable_safety_checker = True

        console.print(
            Panel(
                "The NSFW (not safe for work) checker blurs out images that potentially contain sexual imagery. It can be selectively enabled at run time with --nsfw_checker, and disabled with --no-nsfw_checker.The following option will set whether the checker is enabled by default.Like other options, you can change this setting later by editing the file {self.paths.initfile.location}.NSFW Checker is [bold]NOT[/] recommended for systems with less than 6G VRAM because of the checker's memory requirements."
            )
        )

        if not yes_to_all:
            enable_safety_checker = Confirm.ask(
                "Enable the NSFW checker by default?", default=enable_safety_checker
            )
        else:
            console.print(
                f"Program was started with the --yes switch. NSFW checker is [bold red]{'ON' if enable_safety_checker else 'OFF'}[/]"
            )

        return "--nsfw_checker" if enable_safety_checker else "--no-nsfw_checker"

    def create_initfile(self, **kwargs):
        """
        Create the CLI initialization file
        TODO: template this
        """

        console.print(
            f'Creating the initialization file at "{self.paths.initfile.location}".\n'
        )
        with open(self.paths.initfile.location, "w") as f:
            f.write(
                f"""# InvokeAI initialization file
# This is the InvokeAI initialization file, which contains command-line default values.
# Feel free to edit. If anything goes wrong, you can re-initialize this file by deleting
# or renaming it and then running configure_invokeai.py again.

# the --outdir option controls the default location of image files.
--outdir="{self.paths.outdir.location.expanduser().resolve()}"

# generation arguments
f"{kwargs.get("safety_checker")}"

# You may place other  frequently-used startup commands here, one or more per line.
# Examples:
# --web --host=0.0.0.0
# --steps=20
# -Ak_euler_a -C10.0
#
"""
            )

    def initialize(self, yes_to_all=False):
        """
        Initialize the runtime directory tree
        """

        console.rule(f"Configuring InvokeAI at {self.paths.root.location}")
        console.line()

        if not yes_to_all:
            accepted = False
            while not accepted:
                console.print(
                    f"InvokeAI image outputs will be placed into {self.paths.outdir.location}"
                )
                accepted = Confirm.ask("Accept this location?", default="y")
                if not accepted:
                    self.select_outdir()

        console.print(
            f"You may change the chosen directory at any time by editing --outdir option in {self.paths.initfile.location}. \
            You may also change the runtime directory by setting the environment variable INVOKEAI_ROOT."
        )

        # Create the directory tree
        for location in [
            path.location for path in self.paths.get() if path.kind == "directory"
        ]:
            Path(location).expanduser().absolute().mkdir(exist_ok=True, parents=True)

        # If the default model config file doesn't exist, copy the config dir
        if not self.paths.init_models.location.exists():
            self.copy_configdir()

        self.create_initfile(safety_checker=self.safety_checker_config(yes_to_all))


# class PlainTextInitFile:
#     """
#     Manages the plaintext init file used to configure the launch scripts
#     """

#     def __init__(self, filepath, outputs, safety_checker, sampler, steps) -> None:
#         self.filepath = filepath
#         self.outputs = outputs
#         self.safety_checker = safety_checker
#         self.sampler = sampler
#         self.steps = steps

#     def create(self, filepath: Path):
#         console.print(f"Creating the initialization file at {filepath}")
#         with open(filepath, "w") as f:
#             f.write(
#                 # tempted to template this, but it will be deprecated soon
#                 f"""
# # InvokeAI initialization file
# # This is the InvokeAI initialization file, which contains command-line default values.
# # Feel free to edit. If anything goes wrong, you can re-initialize this file by deleting
# # or renaming it and then running configure_invokeai.py again.

# # the --outdir option controls the default location of image files.
# --outdir="{self.outputs}"

# # generation arguments
# {self.safety_checker}
# --sampler={self.sampler}
# --steps={self.steps}

# # You may place other  frequently-used startup commands here, one or more per line.
# # Examples:
# # --web --host=0.0.0.0
# # --steps=20
# # -Ak_euler_a -C10.0
# #
# """
#             )
