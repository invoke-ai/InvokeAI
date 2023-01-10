"""
InvokeAI Installer
"""

import argparse
from installer import Installer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--destination", type=str, help="Destination path for installation", default="~/invokeai")

    args = parser.parse_args()

    inst = Installer()

    try:
        inst.install(path=args.destination)
    except KeyboardInterrupt as exc:
        print("\n")
        print("Ctrl-C pressed. Aborting.")
        print("See you again soon!")
