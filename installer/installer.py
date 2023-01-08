"""
InvokeAI installer script
"""

import subprocess
import sys
import venv
from pathlib import Path
from tempfile import TemporaryDirectory

SUPPORTED_PYTHON = ">=3.9.0,<3.11"
INSTALLER_REQS = ["rich", "semver"]
PLATFORM = sys.platform

### Feature flags
# Place the virtualenv inside the runtime dir
# (default for 2.2.5) == True
VENV_IN_RUNTIME = True

# Install the wheel from pypi
# (default for 2.2.5) == False
USE_WHEEL = False


class Installer:
    """
    Deploys an InvokeAI installation into a given path
    """

    def __init__(self) -> None:
        self.reqs = INSTALLER_REQS
        self.preflight()
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

    def inst_venv(self) -> TemporaryDirectory:
        """
        Creates a temporary virtual environment for the installer itself

        :return: path to the created virtual environment directory
        :rtype: TemporaryDirectory
        """

        venv_dir = TemporaryDirectory(prefix="invokeai-installer-")
        venv.create(venv_dir.name, with_pip=True)
        sys.path.append(f"{venv_dir.name}/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")

        self.venv_dir = venv_dir
        return venv_dir

    def bootstrap(self, verbose: bool = False) -> TemporaryDirectory:
        """
        Bootstrap the installer venv with packages required at install time

        :return: path to the virtual environment directory that was bootstrapped
        :rtype: TemporaryDirectory
        """

        print("Initializing the installer, please wait...")

        venv_dir = self.inst_venv()
        pip = str(Path(venv_dir.name).absolute() / "bin/pip")

        cmd = [pip, "install", "--require-virtualenv"]
        cmd.extend(self.reqs)

        try:
            res = subprocess.check_output(cmd).decode()
            if verbose:
                print(res)
            return venv_dir
        except subprocess.CalledProcessError as e:
            print(e)

    def install(self, path: str = "~/invokeai", version: str = "latest") -> None:
        """
        Install the InvokeAI application into the given runtime path

        :param path: Destination path for the installation
        :type path: str
        :param version: InvokeAI version to install
        :type version: str
        """

        from messages import dest_path, welcome

        welcome()
        self.dest = dest_path(path)

    def application_venv():
        """
        Create a virtualenv for the InvokeAI installation
        """
        pass


class InvokeAiDeployment:
    """
    Manages an installed instance of InvokeAI
    """

    def __init__(self, path) -> None:
        pass
