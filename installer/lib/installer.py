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
from typing import Union

SUPPORTED_PYTHON = ">=3.9.0,<=3.11.100"
INSTALLER_REQS = ["rich", "semver", "requests", "plumbum", "prompt-toolkit"]
BOOTSTRAP_VENV_PREFIX = "invokeai-installer-tmp"

OS = platform.uname().system
ARCH = platform.uname().machine
VERSION = "latest"

### Feature flags
# Install the virtualenv into the runtime dir
FF_VENV_IN_RUNTIME = True

# Install the wheel packaged with the installer
FF_USE_LOCAL_WHEEL = True


class Installer:
    """
    Deploys an InvokeAI installation into a given path
    """

    def __init__(self) -> None:
        self.reqs = INSTALLER_REQS
        self.preflight()
        if os.getenv("VIRTUAL_ENV") is not None:
            print("A virtual environment is already activated. Please 'deactivate' before installation.")
            sys.exit(-1)
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
            venv_dir = TemporaryDirectory(prefix=BOOTSTRAP_VENV_PREFIX, ignore_cleanup_errors=True)
        else:
            venv_dir = TemporaryDirectory(prefix=BOOTSTRAP_VENV_PREFIX)

        venv.create(venv_dir.name, with_pip=True)
        self.venv_dir = venv_dir
        set_sys_path(Path(venv_dir.name))

        return venv_dir

    def bootstrap(self, verbose: bool = False) -> TemporaryDirectory:
        """
        Bootstrap the installer venv with packages required at install time

        :return: path to the virtual environment directory that was bootstrapped
        :rtype: TemporaryDirectory
        """

        print("Initializing the installer. This may take a minute - please wait...")

        venv_dir = self.mktemp_venv()
        pip = get_pip_from_venv(Path(venv_dir.name))

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

        # Prefer to copy python executables
        # so that updates to system python don't break InvokeAI
        try:
            venv.create(venv_dir, with_pip=True)
        # If installing over an existing environment previously created with symlinks,
        # the executables will fail to copy. Keep symlinks in that case
        except shutil.SameFileError:
            venv.create(venv_dir, with_pip=True, symlinks=True)

        # upgrade pip in Python 3.9 environments
        if int(platform.python_version_tuple()[1]) == 9:
            from plumbum import FG, local

            pip = local[get_pip_from_venv(venv_dir)]
            pip["install", "--upgrade", "pip"] & FG

        return venv_dir

    def install(
        self, root: str = "~/invokeai", version: str = "latest", yes_to_all=False, find_links: Path = None
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

        messages.welcome()

        default_path = os.environ.get("INVOKEAI_ROOT") or Path(root).expanduser().resolve()
        self.dest = default_path if yes_to_all else messages.dest_path(root)

        # create the venv for the app
        self.venv = self.app_venv()

        self.instance = InvokeAiInstance(runtime=self.dest, venv=self.venv, version=version)

        # install dependencies and the InvokeAI application
        (extra_index_url, optional_modules) = get_torch_source() if not yes_to_all else (None, None)
        self.instance.install(
            extra_index_url,
            optional_modules,
            find_links,
        )

        # install the launch/update scripts into the runtime directory
        self.instance.install_user_scripts()

        # run through the configuration flow
        self.instance.configure()


class InvokeAiInstance:
    """
    Manages an installed instance of InvokeAI, comprising a virtual environment and a runtime directory.
    The virtual environment *may* reside within the runtime directory.
    A single runtime directory *may* be shared by multiple virtual environments, though this isn't currently tested or supported.
    """

    def __init__(self, runtime: Path, venv: Path, version: str) -> None:
        self.runtime = runtime
        self.venv = venv
        self.pip = get_pip_from_venv(venv)
        self.version = version

        set_sys_path(venv)
        os.environ["INVOKEAI_ROOT"] = str(self.runtime.expanduser().resolve())
        os.environ["VIRTUAL_ENV"] = str(self.venv.expanduser().resolve())

    def get(self) -> tuple[Path, Path]:
        """
        Get the location of the virtualenv directory for this installation

        :return: Paths of the runtime and the venv directory
        :rtype: tuple[Path, Path]
        """

        return (self.runtime, self.venv)

    def install(self, extra_index_url=None, optional_modules=None, find_links=None):
        """
        Install this instance, including dependencies and the app itself

        :param extra_index_url: the "--extra-index-url ..." line for pip to look in extra indexes.
        :type extra_index_url: str
        """

        import messages

        # install torch first to ensure the correct version gets installed.
        # works with either source or wheel install with negligible impact on installation times.
        messages.simple_banner("Installing PyTorch :fire:")
        self.install_torch(extra_index_url, find_links)

        messages.simple_banner("Installing the InvokeAI Application :art:")
        self.install_app(extra_index_url, optional_modules, find_links)

    def install_torch(self, extra_index_url=None, find_links=None):
        """
        Install PyTorch
        """

        from plumbum import FG, local

        pip = local[self.pip]

        (
            pip[
                "install",
                "--require-virtualenv",
                "numpy~=1.24.0",  # choose versions that won't be uninstalled during phase 2
                "urllib3~=1.26.0",
                "requests~=2.28.0",
                "torch~=2.0.0",
                "torchmetrics==0.11.4",
                "torchvision>=0.14.1",
                "--force-reinstall",
                "--find-links" if find_links is not None else None,
                find_links,
                "--extra-index-url" if extra_index_url is not None else None,
                extra_index_url,
            ]
            & FG
        )

    def install_app(self, extra_index_url=None, optional_modules=None, find_links=None):
        """
        Install the application with pip.
        Supports installation from PyPi or from a local source directory.

        :param extra_index_url: the "--extra-index-url ..." line for pip to look in extra indexes.
        :type extra_index_url: str

        :param optional_modules: optional modules to install using "[module1,module2]" format.
        :type optional_modules: str

        :param find_links: path to a directory containing wheels to be searched prior to going to the internet
        :type find_links: Path
        """

        ## this only applies to pypi installs; TODO actually use this
        if self.version == "pre":
            version = None
            pre = "--pre"
        else:
            version = self.version
            pre = None

        ## TODO: only local wheel will be installed as of now; support for --version arg is TODO
        if FF_USE_LOCAL_WHEEL:
            # if no wheel, try to do a source install before giving up
            try:
                src = str(next(Path(__file__).parent.glob("InvokeAI-*.whl")))
            except StopIteration:
                try:
                    src = Path(__file__).parents[1].expanduser().resolve()
                    # if the above directory contains one of these files, we'll do a source install
                    next(src.glob("pyproject.toml"))
                    next(src.glob("invokeai"))
                except StopIteration:
                    print("Unable to find a wheel or perform a source install. Giving up.")

        elif version == "source":
            # this makes an assumption about the location of the installer package in the source tree
            src = Path(__file__).parents[1].expanduser().resolve()
        else:
            # will install from PyPi
            src = f"invokeai=={version}" if version is not None else "invokeai"

        from plumbum import FG, local

        pip = local[self.pip]

        (
            pip[
                "install",
                "--require-virtualenv",
                "--use-pep517",
                str(src) + (optional_modules if optional_modules else ""),
                "--find-links" if find_links is not None else None,
                find_links,
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

        # set sys.argv to a consistent state
        new_argv = [sys.argv[0]]
        for i in range(1, len(sys.argv)):
            el = sys.argv[i]
            if el in ["-r", "--root"]:
                new_argv.append(el)
                new_argv.append(sys.argv[i + 1])
            elif el in ["-y", "--yes", "--yes-to-all"]:
                new_argv.append(el)
        sys.argv = new_argv

        import requests  # to catch download exceptions
        from messages import introduction

        introduction()

        from invokeai.frontend.install.invokeai_configure import invokeai_configure

        # NOTE: currently the config script does its own arg parsing! this means the command-line switches
        # from the installer will also automatically propagate down to the config script.
        # this may change in the future with config refactoring!
        succeeded = False
        try:
            invokeai_configure()
            succeeded = True
        except requests.exceptions.ConnectionError as e:
            print(f"\nA network error was encountered during configuration and download: {str(e)}")
        except OSError as e:
            print(f"\nAn OS error was encountered during configuration and download: {str(e)}")
        except Exception as e:
            print(f"\nA problem was encountered during the configuration and download steps: {str(e)}")
        finally:
            if not succeeded:
                print('To try again, find the "invokeai" directory, run the script "invoke.sh" or "invoke.bat"')
                print("and choose option 7 to fix a broken install, optionally followed by option 5 to install models.")
                print("Alternatively you can relaunch the installer.")

    def install_user_scripts(self):
        """
        Copy the launch and update scripts to the runtime dir
        """

        ext = "bat" if OS == "Windows" else "sh"

        # scripts = ['invoke', 'update']
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


def get_torch_source() -> (Union[str, None], str):
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

    from messages import graphical_accelerator

    # device can be one of: "cuda", "rocm", "cpu", "idk"
    device = graphical_accelerator()

    url = None
    optional_modules = "[onnx]"
    if OS == "Linux":
        if device == "rocm":
            url = "https://download.pytorch.org/whl/rocm5.4.2"
        elif device == "cpu":
            url = "https://download.pytorch.org/whl/cpu"

    if device == "cuda":
        url = "https://download.pytorch.org/whl/cu118"
        optional_modules = "[xformers,onnx-cuda]"
    if device == "cuda_and_dml":
        url = "https://download.pytorch.org/whl/cu118"
        optional_modules = "[xformers,onnx-directml]"

    # in all other cases, Torch wheels should be coming from PyPi as of Torch 1.13

    return (url, optional_modules)
