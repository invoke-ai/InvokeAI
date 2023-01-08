"""
InvokeAI Installer
"""

from installer import Installer

if __name__ == "__main__":
    inst = Installer()
    try:
        inst.install()
    except KeyboardInterrupt as exc:
        print("\n")
        print("Ctrl-C pressed. Aborting.")
        print("See you again soon!")
