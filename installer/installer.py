"""
InvokeAI installer script
"""

import os
import platform
import subprocess
import sys
import venv
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile
from typing import Union

SUPPORTED_PYTHON = ">=3.9.0,<3.11"
INSTALLER_REQS = ["rich", "semver", "requests", "plumbum", "prompt-toolkit"]

OS = platform.uname().system
ARCH = platform.uname().machine
VERSION = "latest"

### Feature flags
# Install the virtualenv into the runtime dir
FF_VENV_IN_RUNTIME = True

# Install the wheel from pypi
FF_USE_WHEEL = False

INVOKE_AI_SRC = f"https://github.com/invoke-ai/InvokeAI/archive/refs/tags/${VERSION}.zip"


class Installer:
    """
    Deploys an InvokeAI installation into a given path
    """

    def __init__(self) -> None:
        self.reqs = INSTALLER_REQS
        self.preflight()
        if os.getenv("VIRTUAL_ENV") is None:
            # Only bootstrap if not already in a venv
            self.bootstrap()

    def preflight(self) -> None:
        """
        Preflight checks
        """

        # TODO
        # verify python version
        # on macOS verify XCode tools are present
        # verify libmesa, libglx on linux
        # check that the system arch is not i386 (?)
        # check that the system has a GPU, and the type of GPU

        pass

    def mktemp_venv(self) -> TemporaryDirectory:
        """
        Creates a temporary virtual environment for the installer itself

        :return: path to the created virtual environment directory
        :rtype: TemporaryDirectory
        """

        venv_dir = TemporaryDirectory(prefix="invokeai-installer-")
        venv.create(venv_dir.name, with_pip=True)
        self.venv_dir = venv_dir
        add_venv_site(Path(venv_dir.name))

        return venv_dir

    def bootstrap(self, verbose: bool = False) -> TemporaryDirectory:
        """
        Bootstrap the installer venv with packages required at install time

        :return: path to the virtual environment directory that was bootstrapped
        :rtype: TemporaryDirectory
        """

        print("Initializing the installer. This may take a minute - please wait...")

        venv_dir = self.mktemp_venv()
        pip = get_venv_pip(Path(venv_dir.name))

        cmd = [pip, "install", "--require-virtualenv"]
        cmd.extend(self.reqs)

        try:
            res = subprocess.check_output(cmd).decode()
            if verbose:
                print(res)
            return venv_dir
        except subprocess.CalledProcessError as e:
            print(e)

    def app_venv(self, path: str = None):
        """
        Create a virtualenv for the InvokeAI installation
        """

        # explicit venv location
        # currently unused in normal operation
        # useful for testing or special cases
        if path is not None:
            venv_dir = Path(path)

        # experimental / testing
        #
        elif not FF_VENV_IN_RUNTIME:
            if OS == "Windows":
                venv_dir_parent = os.getenv("APPDATA", "~/AppData/Roaming")
            elif OS == "Darwin":
                # there is no environment variable on macOS to find this
                # TODO: confirm this is working as expected
                venv_dir_parent = "~/Library/Application Support"
            elif OS == "Linux":
                venv_dir_parent = os.getenv("XDG_DATA_DIR", "~/.local/share")
            venv_dir = Path(venv_dir_parent).expanduser().resolve() / f"InvokeAI/{VERSION}/venv"

        # stable / current
        else:
            venv_dir = self.dest / ".venv"

        venv.create(venv_dir, with_pip=True)
        return venv_dir

    def get_payload():
        """
        Obtain the InvokeAI installation payload
        """

        pass

    def install(self, path: str = "~/invokeai", version: str = "latest") -> None:
        """
        Install the InvokeAI application into the given runtime path

        :param path: Destination path for the installation
        :type path: str
        :param version: InvokeAI version to install
        :type version: str
        """

        import messages

        messages.welcome()

        self.dest = messages.dest_path(path)

        self.venv = self.app_venv()

        self.instance = InvokeAiInstance(runtime=self.dest, venv=self.venv)

        self.instance.deploy(extra_index_url=get_torch_source())

        self.instance.configure()


class InvokeAiInstance:
    """
    Manages an installed instance of InvokeAI, comprising a virtual environment and a runtime directory.
    The virtual environment *may* reside within the runtime directory.
    A single runtime directory *may* be shared by multiple virtual environments, though this isn't currently tested or supported.
    """

    def __init__(self, runtime: Path, venv: Path) -> None:

        self.runtime = runtime
        self.venv = venv
        self.pip = get_venv_pip(venv)

        add_venv_site(venv)
        os.environ["INVOKEAI_ROOT"] = str(self.runtime.expanduser().resolve())
        os.environ["VIRTUAL_ENV"] = str(self.venv.expanduser().resolve())

    def get(self) -> tuple[Path, Path]:
        """
        Get the location of the virtualenv directory for this installation

        :return: Paths of the runtime and the venv directory
        :rtype: tuple[Path, Path]
        """

        return (self.runtime, self.venv)

    def deploy(self, extra_index_url=None):
        """
        Install packages with pip

        :param extra_index_url: the "--extra-index-url ..." line for pip to look in extra indexes.
        :type extra_index_url: str
        """

        ### this is all very rough for now as a PoC
        ### source installer basically
        ### TODO: need to pull the source from Github like the current installer does
        ### until we continuously build wheels

        import messages
        from plumbum import local, FG

        # pre-installing Torch because this is the most reliable way to ensure
        # the correct version gets installed.
        # this works with either source or wheel install and has
        # negligible impact on installation times.
        messages.simple_banner("Installing PyTorch :fire:")
        self.install_torch(extra_index_url)

        messages.simple_banner("Installing InvokeAI base dependencies :rocket:")
        extra_index_url_arg = "--extra-index-url" if extra_index_url is not None else None

        pip = local[self.pip]

        (
            pip[
                "install",
                "--require-virtualenv",
                "-r",
                (Path(__file__).parents[1] / "environments-and-requirements/requirements-base.txt")
                .expanduser()
                .resolve(),
                extra_index_url_arg,
                extra_index_url,
            ]
            & FG
        )

        messages.simple_banner("Installing the InvokeAI Application :art:")
        (
            pip[
                "install",
                "--require-virtualenv",
                Path(__file__).parents[1].expanduser().resolve(),
                extra_index_url_arg,
                extra_index_url,
            ]
            & FG
        )

    def install_torch(self, extra_index_url=None):
        """
        Install PyTorch
        """

        from plumbum import local, FG

        extra_index_url_arg = "--extra-index-url" if extra_index_url is not None else None

        pip = local[self.pip]

        (
            pip[
                "install",
                "--require-virtualenv",
                "torch",
                "torchvision",
                extra_index_url_arg,
                extra_index_url,
            ]
            & FG
        )

    def configure(self):
        """
        Configure the InvokeAI runtime directory
        """

        from messages import introduction

        introduction()

        from ldm.invoke.config import configure_invokeai

        configure_invokeai.main()

    def update(self):
        pass

    def remove(self):
        pass


### Utility functions ###


def get_venv_pip(venv_path: Path) -> str:
    """
    Given a path to a virtual environment, get the absolute path to the `pip` executable
    in a cross-platform fashion. Does not validate that the pip executable
    actually exists in the virtualenv.

    :param venv_path: Path to the virtual environment
    :type venv_path: Path
    :return: Absolute path to the pip executable
    :rtype: str
    """

    pip = "Scripts\pip.exe" if OS == "Windows" else "bin/pip"
    return str(venv_path.absolute() / pip)


def add_venv_site(venv_path: Path) -> None:
    """
    Given a path to a virtual environment, add the python site-packages directory from this venv
    into the sys.path, in a cross-platform fashion, such that packages from this venv
    may be imported in the current process.

    :param venv_path: Path to the virtual environment
    :type venv_path: Path
    """

    lib = "Lib" if OS == "Windows" else f"lib/python{sys.version_info.major}.{sys.version_info.minor}"
    sys.path.append(str(Path(venv_path, lib, "site-packages").absolute()))


def get_torch_source() -> Union[str, None]:
    """
    Determine the extra index URL for pip to use for torch installation.
    This depends on the OS and the graphics accelerator in use.
    This is only applicable to Windows and Linux, since PyTorch does not
    offer accelerated builds for macOS.

    Prefer CUDA if the user wasn't sure of their GPU, as it will fallback to CPU if possible.

    A NoneType return means just go to PyPi.

    :return: The list of arguments to pip pointing at the PyTorch wheel source, if available
    :rtype: list
    """

    from messages import graphical_accelerator

    device = graphical_accelerator()

    url = None
    if OS == "Linux":
        if device in ["cuda", "idk"]:
            url = "https://download.pytorch.org/whl/cu117"
        elif device == "rocm":
            url = "https://download.pytorch.org/whl/rocm5.2"
        else:
            url = "https://download.pytorch.org/whl/cpu"

    elif OS == "Windows":
        if device in ["cuda", "idk"]:
            url = "https://download.pytorch.org/whl/cu117"

    # ignoring macOS because its wheels come from PyPi anyway (cpu only)

    return url
