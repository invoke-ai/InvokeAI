"""
InvokeAI Installer
"""

import argparse
import os
from pathlib import Path
from installer import Installer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-r",
        "--root",
        dest="root",
        type=str,
        help="Destination path for installation",
        default=os.environ.get("INVOKEAI_ROOT") or "~/invokeai",
    )
    parser.add_argument(
        "-y",
        "--yes",
        "--yes-to-all",
        dest="yes_to_all",
        action="store_true",
        help="Assume default answers to all questions",
        default=False,
    )

    parser.add_argument(
        "--version",
        dest="version",
        help="Version of InvokeAI to install. Default to the latest stable release. A special 'pre' value will install the latest published pre-release version.",
        default=None,
    )

    parser.add_argument(
        "--find-links",
        dest="find_links",
        help="Specifies a directory of local wheel files to be searched prior to searching the online repositories.",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    inst = Installer()

    try:
        inst.install(**args.__dict__)
    except KeyboardInterrupt:
        print("\n")
        print("Ctrl-C pressed. Aborting.")
        print("Come back soon!")
