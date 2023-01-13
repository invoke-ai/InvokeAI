"""
InvokeAI Installer
"""

import argparse
from installer import Installer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--root", type=str, help="Destination path for installation", default="~/invokeai")

    args = parser.parse_args()

    inst = Installer()

    try:
        inst.install(path=args.root)
    except KeyboardInterrupt as exc:
        print("\n")
        print("Ctrl-C pressed. Aborting.")
        print("Come back soon!")
