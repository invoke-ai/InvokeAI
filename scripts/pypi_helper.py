import requests as request

import ldm.invoke._version as version

local_version = str(version.__version__)


def get_pypi_versions(package_name="InvokeAI") -> list[str]:
    """Get the versions of the package from PyPI"""
    url = f"https://pypi.org/pypi/{package_name}/json"
    response = request.get(url).json()
    versions: list[str] = list(response["releases"].keys())
    return versions


def local_on_pypi(package_name="InvokeAI", local_version=local_version) -> bool:
    """Compare the versions of the package from PyPI and the local package"""
    pypi_versions = get_pypi_versions(package_name)
    return local_version in pypi_versions


if __name__ == "__main__":
    package_name = "InvokeAI"
    if local_on_pypi():
        print(f"Package {package_name} is up to date")
    else:
        print(f"Package {package_name} is not up to date")
