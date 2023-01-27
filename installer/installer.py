"""
InvokeAI installer script
"""

import os
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from tempfile import TemporaryDirectory
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

        # Cleaning up temporary directories on Windows results in a race condition
        # and a stack trace.
        # `ignore_cleanup_errors` was only added in Python 3.10
        # users of Python 3.9 will see a gnarly stack trace on installer exit
        if OS == "Windows" and int(platform.python_version_tuple()[1]) >= 10:
            venv_dir = TemporaryDirectory(prefix="invokeai-installer-", ignore_cleanup_errors=True)
        else:
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

        cmd = [pip, "install", "--require-virtualenv", "--use-pep517"]
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

    def install(self, root: str = "~/invokeai", version: str = "latest", yes_to_all=False) -> None:
        """
        Install the InvokeAI application into the given runtime path

        :param root: Destination path for the installation
        :type root: str
        :param version: InvokeAI version to install
        :type version: str
        :param yes: Accept defaults to all questions
        :type yes: bool
        """

        import messages

        messages.welcome()

        self.dest = Path(root).expanduser().resolve() if yes_to_all else messages.dest_path(root)

        # create the venv for the app
        self.venv = self.app_venv()

        self.instance = InvokeAiInstance(runtime=self.dest, venv=self.venv, version=version)

        # install dependencies and the InvokeAI application

        self.instance.install(extra_index_url = get_torch_source() if not yes_to_all else None)

        # run through the configuration flow
        self.instance.configure()

        # install the launch/update scripts into the runtime directory
        self.instance.install_user_scripts()


class InvokeAiInstance:
    """
    Manages an installed instance of InvokeAI, comprising a virtual environment and a runtime directory.
    The virtual environment *may* reside within the runtime directory.
    A single runtime directory *may* be shared by multiple virtual environments, though this isn't currently tested or supported.
    """

    def __init__(self, runtime: Path, venv: Path, version: str) -> None:

        self.runtime = runtime
        self.venv = venv
        self.pip = get_venv_pip(venv)
        self.version = version

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

    def install(self, extra_index_url=None):
        """
        Install this instance, including depenencies and the app itself

        :param extra_index_url: the "--extra-index-url ..." line for pip to look in extra indexes.
        :type extra_index_url: str
        """

        import messages

        # install torch first to ensure the correct version gets installed.
        # works with either source or wheel install with negligible impact on installation times.
        messages.simple_banner("Installing PyTorch :fire:")
        self.install_torch(extra_index_url)

        messages.simple_banner("Installing the InvokeAI Application :art:")
        self.install_app(extra_index_url)

    def install_torch(self, extra_index_url=None):
        """
        Install PyTorch
        """

        from plumbum import FG, local

        pip = local[self.pip]

        (
            pip[
                "install",
                "--require-virtualenv",
                "torch",
                "torchvision",
                "--extra-index-url" if extra_index_url is not None else None,
                extra_index_url,
            ]
            & FG
        )

    def install_app(self, extra_index_url=None):
        """
        Install the application with pip.
        Supports installation from PyPi or from a local source directory.

        :param extra_index_url: the "--extra-index-url ..." line for pip to look in extra indexes.
        :type extra_index_url: str
        """

        if self.version == "pre":
            version = None
            pre = "--pre"
        else:
            version = self.version
            pre = None

        if FF_USE_WHEEL:
            src = f"invokeai=={version}" if version is not None else "invokeai"
        else:
            # this makes an assumption about the location of the installer package in the source tree
            src = Path(__file__).parents[1].expanduser().resolve()

        import messages
        from plumbum import FG, local

        pip = local[self.pip]

        (
            pip[
                "install",
                "--require-virtualenv",
                "--use-pep517",
                src,
                "--extra-index-url" if extra_index_url is not None else None,
                extra_index_url,
                pre,
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

        # NOTE: currently the config script does its own arg parsing! this means the command-line switches
        # from the installer will also automatically propagate down to the config script.
        # this may change in the future with config refactoring!
        configure_invokeai.main()

    def install_user_scripts(self):
        """
        Copy the launch and update scripts to the runtime dir
        """

        ext = "bat" if OS == "Windows" else "sh"

        for script in ["invoke", "update"]:
            src = Path(__file__).parent / "templates" / f"{script}.{ext}.in"
            dest = self.runtime / f"{script}.{ext}"
            shutil.copy(src, dest)
            os.chmod(dest, 0o0755)

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
    return str(venv_path.expanduser().resolve() / pip)


def add_venv_site(venv_path: Path) -> None:
    """
    Given a path to a virtual environment, add the python site-packages directory from this venv
    into the sys.path, in a cross-platform fashion, such that packages from this venv
    may be imported in the current process.

    :param venv_path: Path to the virtual environment
    :type venv_path: Path
    """

    lib = "Lib" if OS == "Windows" else f"lib/python{sys.version_info.major}.{sys.version_info.minor}"
    sys.path.append(str(Path(venv_path, lib, "site-packages").expanduser().resolve()))


def get_torch_source() -> Union[str, None]:
    """
    Determine the extra index URL for pip to use for torch installation.
    This depends on the OS and the graphics accelerator in use.
    This is only applicable to Windows and Linux, since PyTorch does not
    offer accelerated builds for macOS.

    Prefer CUDA-enabled wheels if the user wasn't sure of their GPU, as it will fallback to CPU if possible.

    A NoneType return means just go to PyPi.

    :return: The list of arguments to pip pointing at the PyTorch wheel source, if available
    :rtype: list
    """

    from messages import graphical_accelerator

    # device can be one of: "cuda", "rocm", "cpu", "idk"
    device = graphical_accelerator()

    url = None
    if OS == "Linux":
        if device == "rocm":
            url = "https://download.pytorch.org/whl/rocm5.2"
        elif device == "cpu":
            url = "https://download.pytorch.org/whl/cpu"

    # in all other cases, Torch wheels should be coming from PyPi as of Torch 1.13

    return url
