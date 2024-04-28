"""
- Checks GitHub releases for updates, displaying any updates
- If the user chooses to update, downloads the latest installer and unzips it to the dir provide as an arg
"""

import argparse
import re
import shutil
from pathlib import Path

import requests
from packaging.version import Version

from invokeai.version.invokeai_version import __version__ as invokeai_version


def check_for_updates(install_temp_dir: str) -> None:
    current_version = Version(invokeai_version)
    print(f"Current version: v{current_version}")
    print("Checking for updates... ", end="")  # no trailing newline

    response = requests.get("https://api.github.com/repos/invoke-ai/InvokeAI/releases")
    response.raise_for_status()
    req_json = response.json()

    latest_release = [r for r in req_json if not r["prerelease"]][0]
    latest_release_version = Version(latest_release["tag_name"])
    is_release_available = current_version < latest_release_version

    latest_prerelease = [r for r in req_json if r["prerelease"]][0]
    latest_prerelease_version = Version(latest_prerelease["tag_name"])
    is_prerelease_available = current_version < latest_prerelease_version

    if not is_release_available and not is_prerelease_available:
        print("up to date!")
    else:
        print("updates available:")
        if is_release_available:
            print(f"- New release: v{latest_release_version}")
        if is_prerelease_available:
            print(f"- New pre-release: v{latest_prerelease_version}")

    prompt_msg = "\nDo you want to run the updater? You may select a version to update to or reinstall the current version. (y/n) [y]: "
    response = input(prompt_msg).lower()

    if response == "" or response == "y":
        # Get the installer asset from the latest release - the asset name is InvokeAI-installer-*.zip
        # TODO: It's possible that the prerelease installer is different from the release installer, but in an effort
        # to keep this simple, we will always use the release installer...
        installer_asset = next(
            (a for a in latest_release["assets"] if re.match(r"InvokeAI-installer.*zip", a["name"], re.IGNORECASE))
        )
        download_url = installer_asset["browser_download_url"]
        # Download & unzip the installer (it's only a few KB)
        install_temp_path = Path(install_temp_dir, "installer.zip")
        install_temp_path.parent.mkdir(exist_ok=True)
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        with open(install_temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        shutil.unpack_archive(install_temp_path, install_temp_path.parent)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("install_temp_dir", type=str, help="Path to write the release URL to")
    args = parser.parse_args()
    check_for_updates(args.install_temp_dir)


if __name__ == "__main__":
    main()
