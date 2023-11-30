#!/usr/bin/env python

import requests

from invokeai.version import __version__

local_version = str(__version__).replace("-", "")
package_name = "InvokeAI"


def get_pypi_versions(package_name=package_name) -> list[str]:
    """Get the versions of the package from PyPI"""
    url = f"https://pypi.org/pypi/{package_name}/json"
    response = requests.get(url).json()
    versions: list[str] = list(response["releases"].keys())
    return versions


def local_on_pypi(package_name=package_name, local_version=local_version) -> bool:
    """Compare the versions of the package from PyPI and the local package"""
    pypi_versions = get_pypi_versions(package_name)
    return local_version in pypi_versions


if __name__ == "__main__":
    if local_on_pypi():
        print(f"Package {package_name} is up to date")
    else:
        print(f"Package {package_name} is not up to date")
