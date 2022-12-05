# from omegaconf import OmegaConf
from dataclasses import dataclass
from pathlib import Path
import os

from .globals import Globals
from .paths import Paths

try:
    from rich.console import Console
    from rich.prompt import Prompt, Confirm
    console = Console()
    print = console.print
    line = console.line
except Exception:
    line = print("\n")

@dataclass
class Config:

    try_patchmatch = Globals.try_patchmatch

    enable_safety_checker = True
    default_sampler = 'k_heun'
    default_steps = '20'  # deliberately a string - see test below

    sampler_choices = ['ddim','k_dpm_2_a','k_dpm_2','k_euler_a','k_euler','k_heun','k_lms','plms']

    # future
    user_weights_dirs = []

class RuntimeDir:
    """
    Manages the runtime directory
    """

    def __init__(self, root_path: str = None) -> None:

        self.config = Config()
        self.paths = Paths()

        # for convenience
        self.root = self.paths.root.location
        self.outputs = self.paths.outputs.location

    def set_root(self, root_path: str = None) -> Path:

        if root_path is None:
            if (root_path := os.getenv("INVOKEAI_ROOT")) is not None:
                print(f"INVOKEAI_ROOT={root_path} environment variable is set")

        self.paths.set_root(root_path)
        self.root = self.paths.root.location

        print(f"Using {Path(self.root).expanduser().resolve()} as the InvokeAI runtime directory")
        return self.root

    def set_outputs(self, output_path: str = None) -> Path:

        if output_path is None:
            # Check env var
            ### TODO NOT YET DOCUMENTED
            if (output_path := os.getenv("INVOKEAI_OUTPUTS")) is not None:
                print(f"Found INVOKEAI_OUTPUTS={output_path} environment variable")


        ### TODO THIS DOES NOT TAKE EFFECT FOR SOME REASON
        ### location remains unchanged
        self.paths.outputs.location = Path(output_path)
        self.outputs = self.paths.root.location / output_path

        print(f"Generated images will be placed under {Path(self.outputs).expanduser().resolve()}")
        return self.outputs

    def validate(self) -> bool:
        """
        Validate that the runtime dir is correctly configured
        """

        console.rule(f"Validating the runtime directory at {self.root.expanduser().resolve()}")
        console.line()

        missing = False
        for this in self.paths.get():
            abspath = this.location.expanduser().resolve()
            if abspath.exists():
                msg_prefix = ":white_check_mark:"
            else:
                msg_prefix = ":x:"
                missing = True
            print(f"{msg_prefix} {this.description} {this.kind}: {abspath}")

        return not missing

    def initialize(self, yes_to_all=False):
        """
        Initialize the runtime directory tree
        """

        print(f"[bold]Initializing invokeai runtime directory at {self.root}...")



        # console.print()

        # print(f'\nYou may change the chosen directories at any time by editing the --root and --outdir options in "{Globals.initfile}",')
        # print(f'You may also change the runtime directory by setting the environment variable INVOKEAI_ROOT.\n')


        # if not yes_to_all:
        #     print('The NSFW (not safe for work) checker blurs out images that potentially contain sexual imagery.')
        #     print('It can be selectively enabled at run time with --nsfw_checker, and disabled with --no-nsfw_checker.')
        #     print('The following option will set whether the checker is enabled by default. Like other options, you can')
        #     print(f'change this setting later by editing the file {Globals.initfile}.')
        #     enable_safety_checker = yes_or_no('Enable the NSFW checker by default?',enable_safety_checker)

        #     print('\nThe next choice selects the sampler to use by default. Samplers have different speed/performance')
        #     print('tradeoffs. If you are not sure what to select, accept the default.')
        #     sampler = None
        #     while sampler not in sampler_choices:
        #         sampler = input(f'Default sampler to use? ({", ".join(sampler_choices)}) [{default_sampler}]:') or default_sampler

        #     print('\nThe number of denoising steps affects both the speed and quality of the images generated.')
        #     print('Higher steps often (but not always) increases the quality of the image, but increases image')
        #     print('generation time. This can be changed at run time. Accept the default if you are unsure.')
        #     steps = ''
        #     while not steps.isnumeric():
        #         steps = input(f'Default number of steps to use during generation? [{default_steps}]:') or default_steps
        # else:
        #     sampler = default_sampler
        #     steps = default_steps

        # safety_checker = '--nsfw_checker' if enable_safety_checker else '--no-nsfw_checker'

        # for name in ('models','configs','embeddings'):
        #     os.makedirs(os.path.join(root,name), exist_ok=True)
        # for src in (['configs']):
        #     dest = os.path.join(root,src)
        #     if not os.path.samefile(src,dest):
        #         shutil.copytree(src,dest,dirs_exist_ok=True)
        #     os.makedirs(outputs, exist_ok=True)

        # init_file = os.path.expanduser(Globals.initfile)


    #-------------------------------------


class PlainTextInitFile():
    """
    Manages the plaintext init file used to configure the launch scripts
    """

    def __init__(self, filepath, outputs, safety_checker, sampler, steps) -> None:
        self.filepath = filepath
        self.outputs = outputs
        self.safety_checker = safety_checker
        self.sampler = sampler
        self.steps = steps

    def create(self, filepath: Path):
        console.print(f"Creating the initialization file at {filepath}")
        with open(filepath,"w") as f:
            f.write(
# tempted to template this, but it will be deprecated soon
f'''
# InvokeAI initialization file
# This is the InvokeAI initialization file, which contains command-line default values.
# Feel free to edit. If anything goes wrong, you can re-initialize this file by deleting
# or renaming it and then running configure_invokeai.py again.

# the --outdir option controls the default location of image files.
--outdir="{self.outputs}"

# generation arguments
{self.safety_checker}
--sampler={self.sampler}
--steps={self.steps}

# You may place other  frequently-used startup commands here, one or more per line.
# Examples:
# --web --host=0.0.0.0
# --steps=20
# -Ak_euler_a -C10.0
#
''')
