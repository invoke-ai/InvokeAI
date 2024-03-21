# Copyright (c) 2023 Eugene Brodsky (https://github.com/ebr)
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
from typing import Optional, Tuple

SUPPORTED_PYTHON = ">=3.10.0,<=3.11.100"
INSTALLER_REQS = ["rich", "semver", "requests", "plumbum", "prompt-toolkit"]
BOOTSTRAP_VENV_PREFIX = "invokeai-installer-tmp"

OS = platform.uname().system
ARCH = platform.uname().machine
VERSION = "latest"


class Installer:
    """
    Deploys an InvokeAI installation into a given path
    """

    reqs: list[str] = INSTALLER_REQS

    def __init__(self) -> None:
        if os.getenv("VIRTUAL_ENV") is not None:
            print("A virtual environment is already activated. Please 'deactivate' before installation.")
            sys.exit(-1)
        self.bootstrap()
        self.available_releases = get_github_releases()

    def mktemp_venv(self) -> TemporaryDirectory:
        """
        Creates a temporary virtual environment for the installer itself

        :return: path to the created virtual environment directory
        :rtype: TemporaryDirectory
        """

        # Cleaning up temporary directories on Windows results in a race condition
        # and a stack trace.
        # `ignore_cleanup_errors` was only added in Python 3.10
        if OS == "Windows" and int(platform.python_version_tuple()[1]) >= 10:
            venv_dir = TemporaryDirectory(prefix=BOOTSTRAP_VENV_PREFIX, ignore_cleanup_errors=True)
        else:
            venv_dir = TemporaryDirectory(prefix=BOOTSTRAP_VENV_PREFIX)

        venv.create(venv_dir.name, with_pip=True)
        self.venv_dir = venv_dir
        set_sys_path(Path(venv_dir.name))

        return venv_dir

    def bootstrap(self, verbose: bool = False) -> TemporaryDirectory | None:
        """
        Bootstrap the installer venv with packages required at install time
        """

        print("Initializing the installer. This may take a minute - please wait...")

        venv_dir = self.mktemp_venv()
        pip = get_pip_from_venv(Path(venv_dir.name))

        cmd = [pip, "install", "--require-virtualenv", "--use-pep517"]
        cmd.extend(self.reqs)

        try:
            # upgrade pip to the latest version to avoid a confusing message
            res = upgrade_pip(Path(venv_dir.name))
            if verbose:
                print(res)

            # run the install prerequisites installation
            res = subprocess.check_output(cmd).decode()

            if verbose:
                print(res)

            return venv_dir
        except subprocess.CalledProcessError as e:
            print(e)

    def app_venv(self, venv_parent) -> Path:
        """
        Create a virtualenv for the InvokeAI installation
        """

        venv_dir = venv_parent / ".venv"

        # Prefer to copy python executables
        # so that updates to system python don't break InvokeAI
        try:
            venv.create(venv_dir, with_pip=True)
        # If installing over an existing environment previously created with symlinks,
        # the executables will fail to copy. Keep symlinks in that case
        except shutil.SameFileError:
            venv.create(venv_dir, with_pip=True, symlinks=True)

        return venv_dir

    def install(
        self, version=None, root: str = "~/invokeai", yes_to_all=False, find_links: Optional[Path] = None
    ) -> None:
        """
        Install the InvokeAI application into the given runtime path

        :param root: Destination path for the installation
        :type root: str
        :param version: InvokeAI version to install
        :type version: str
        :param yes: Accept defaults to all questions
        :type yes: bool
        :param find_links: A local directory to search for requirement wheels before going to remote indexes
        :type find_links: Path
        """

        import messages

        messages.welcome(self.available_releases)

        version = messages.choose_version(self.available_releases)

        auto_dest = Path(os.environ.get("INVOKEAI_ROOT", root)).expanduser().resolve()
        destination = auto_dest if yes_to_all else messages.dest_path(root)
        if destination is None:
            print("Could not find or create the destination directory. Installation cancelled.")
            sys.exit(0)

        # create the venv for the app
        self.venv = self.app_venv(venv_parent=destination)

        self.instance = InvokeAiInstance(runtime=destination, venv=self.venv, version=version)

        # install dependencies and the InvokeAI application
        (extra_index_url, optional_modules) = get_torch_source() if not yes_to_all else (None, None)
        self.instance.install(
            extra_index_url,
            optional_modules,
            find_links,
        )

        # install the launch/update scripts into the runtime directory
        self.instance.install_user_scripts()


class InvokeAiInstance:
    """
    Manages an installed instance of InvokeAI, comprising a virtual environment and a runtime directory.
    The virtual environment *may* reside within the runtime directory.
    A single runtime directory *may* be shared by multiple virtual environments, though this isn't currently tested or supported.
    """

    def __init__(self, runtime: Path, venv: Path, version: str = "stable") -> None:
        self.runtime = runtime
        self.venv = venv
        self.pip = get_pip_from_venv(venv)
        self.version = version

        set_sys_path(venv)
        os.environ["INVOKEAI_ROOT"] = str(self.runtime.expanduser().resolve())
        os.environ["VIRTUAL_ENV"] = str(self.venv.expanduser().resolve())
        upgrade_pip(venv)

    def get(self) -> tuple[Path, Path]:
        """
        Get the location of the virtualenv directory for this installation

        :return: Paths of the runtime and the venv directory
        :rtype: tuple[Path, Path]
        """

        return (self.runtime, self.venv)

    def install(self, extra_index_url=None, optional_modules=None, find_links=None):
        """
        Install the package from PyPi.

        :param extra_index_url: the "--extra-index-url ..." line for pip to look in extra indexes.
        :type extra_index_url: str

        :param optional_modules: optional modules to install using "[module1,module2]" format.
        :type optional_modules: str

        :param find_links: path to a directory containing wheels to be searched prior to going to the internet
        :type find_links: Path
        """

        import messages

        # not currently used, but may be useful for "install most recent version" option
        if self.version == "prerelease":
            version = None
            pre_flag = "--pre"
        elif self.version == "stable":
            version = None
            pre_flag = None
        else:
            version = self.version
            pre_flag = None

        src = "invokeai"
        if optional_modules:
            src += optional_modules
        if version:
            src += f"=={version}"

        messages.simple_banner("Installing the InvokeAI Application :art:")

        from plumbum import FG, ProcessExecutionError, local  # type: ignore

        pip = local[self.pip]

        pipeline = pip[
            "install",
            "--require-virtualenv",
            "--force-reinstall",
            "--use-pep517",
            str(src),
            "--find-links" if find_links is not None else None,
            find_links,
            "--extra-index-url" if extra_index_url is not None else None,
            extra_index_url,
            pre_flag,
        ]

        try:
            _ = pipeline & FG
        except ProcessExecutionError as e:
            print(f"Error: {e}")
            print(
                "Could not install InvokeAI. Please try downloading the latest version of the installer and install again."
            )
            sys.exit(1)

    def install_user_scripts(self):
        """
        Copy the launch and update scripts to the runtime dir
        """

        ext = "bat" if OS == "Windows" else "sh"

        scripts = ["invoke"]

        for script in scripts:
            src = Path(__file__).parent / ".." / "templates" / f"{script}.{ext}.in"
            dest = self.runtime / f"{script}.{ext}"
            shutil.copy(src, dest)
            os.chmod(dest, 0o0755)

    def update(self):
        pass

    def remove(self):
        pass


### Utility functions ###


def get_pip_from_venv(venv_path: Path) -> str:
    """
    Given a path to a virtual environment, get the absolute path to the `pip` executable
    in a cross-platform fashion. Does not validate that the pip executable
    actually exists in the virtualenv.

    :param venv_path: Path to the virtual environment
    :type venv_path: Path
    :return: Absolute path to the pip executable
    :rtype: str
    """

    pip = "Scripts\\pip.exe" if OS == "Windows" else "bin/pip"
    return str(venv_path.expanduser().resolve() / pip)


def upgrade_pip(venv_path: Path) -> str | None:
    """
    Upgrade the pip executable in the given virtual environment
    """

    python = "Scripts\\python.exe" if OS == "Windows" else "bin/python"
    python = str(venv_path.expanduser().resolve() / python)

    try:
        result = subprocess.check_output([python, "-m", "pip", "install", "--upgrade", "pip"]).decode()
    except subprocess.CalledProcessError as e:
        print(e)
        result = None

    return result


def set_sys_path(venv_path: Path) -> None:
    """
    Given a path to a virtual environment, set the sys.path, in a cross-platform fashion,
    such that packages from the given venv may be imported in the current process.
    Ensure that the packages from system environment are not visible (emulate
    the virtual env 'activate' script) - this doesn't work on Windows yet.

    :param venv_path: Path to the virtual environment
    :type venv_path: Path
    """

    # filter out any paths in sys.path that may be system- or user-wide
    # but leave the temporary bootstrap virtualenv as it contains packages we
    # temporarily need at install time
    sys.path = list(filter(lambda p: not p.endswith("-packages") or p.find(BOOTSTRAP_VENV_PREFIX) != -1, sys.path))

    # determine site-packages/lib directory location for the venv
    lib = "Lib" if OS == "Windows" else f"lib/python{sys.version_info.major}.{sys.version_info.minor}"

    # add the site-packages location to the venv
    sys.path.append(str(Path(venv_path, lib, "site-packages").expanduser().resolve()))


def get_github_releases() -> tuple[list, list] | None:
    """
    Query Github for published (pre-)release versions.
    Return a tuple where the first element is a list of stable releases and the second element is a list of pre-releases.
    Return None if the query fails for any reason.
    """

    import requests

    ## get latest releases using github api
    url = "https://api.github.com/repos/invoke-ai/InvokeAI/releases"
    releases, pre_releases = [], []
    try:
        res = requests.get(url)
        res.raise_for_status()
        tag_info = res.json()
        for tag in tag_info:
            if not tag["prerelease"]:
                releases.append(tag["tag_name"].lstrip("v"))
            else:
                pre_releases.append(tag["tag_name"].lstrip("v"))
    except requests.HTTPError as e:
        print(f"Error: {e}")
        print("Could not fetch version information from GitHub. Please check your network connection and try again.")
        return
    except Exception as e:
        print(f"Error: {e}")
        print("An unexpected error occurred while trying to fetch version information from GitHub. Please try again.")
        return

    releases.sort(reverse=True)
    pre_releases.sort(reverse=True)

    return releases, pre_releases


def get_torch_source() -> Tuple[str | None, str | None]:
    """
    Determine the extra index URL for pip to use for torch installation.
    This depends on the OS and the graphics accelerator in use.
    This is only applicable to Windows and Linux, since PyTorch does not
    offer accelerated builds for macOS.

    Prefer CUDA-enabled wheels if the user wasn't sure of their GPU, as it will fallback to CPU if possible.

    A NoneType return means just go to PyPi.

    :return: tuple consisting of (extra index url or None, optional modules to load or None)
    :rtype: list
    """

    from messages import select_gpu

    # device can be one of: "cuda", "rocm", "cpu", "cuda_and_dml, autodetect"
    device = select_gpu()

    url = None
    optional_modules = "[onnx]"
    if OS == "Linux":
        if device.value == "rocm":
            url = "https://download.pytorch.org/whl/rocm5.6"
        elif device.value == "cpu":
            url = "https://download.pytorch.org/whl/cpu"

    elif OS == "Windows":
        if device.value == "cuda":
            url = "https://download.pytorch.org/whl/cu121"
            optional_modules = "[xformers,onnx-cuda]"
        if device.value == "cuda_and_dml":
            url = "https://download.pytorch.org/whl/cu121"
            optional_modules = "[xformers,onnx-directml]"

    # in all other cases, Torch wheels should be coming from PyPi as of Torch 1.13

    return (url, optional_modules)
