"""
script to setup the current working directory as the invokeai_root
(unless Globals.root is set) will not overwrite existing directorys
"""
import os
import shutil
import sys

from ldm.invoke.globals import Globals


def main():
    """
    main function
    """
    osp = os.path
    source_prefix = os.path.join(
        sys.prefix,
        sys.platlibdir,
        os.listdir(os.path.join(sys.prefix, sys.platlibdir))[0],
        "site-packages",
    )
    dirs_to_copy = "configs", "frontend", "scripts"
    done: int = 0

    def copy_from_venv(folder):
        """
        function to copy the requested directory tree from source_prefix to Globals.root
        usefull since the config script will not work in a empty directory.
        """
        # src_folder = osp.abspath(osp.join(os.environ.get("VIRTUAL_ENV"), folder))
        src_folder = osp.abspath(osp.join(source_prefix, folder))
        if Globals.root:
            dest_folder = Globals.root
        else:
            dest_folder = osp.abspath(
                osp.join(os.environ.get("VIRTUAL_ENV"), "..", folder)
            )
        # dest_folder = osp.join(Globals.root, folder)
        print(f"trying to copy {src_folder} to {dest_folder}")
        shutil.copytree(src_folder, dest_folder)

    for tree in dirs_to_copy:
        try:
            copy_from_venv(folder=tree)
            done = done + 1
        except FileExistsError as err:
            print(err)
        except FileNotFoundError as err:
            print(err)
        except RuntimeError as err:
            print(err)

    if int(len(dirs_to_copy)) == done:
        from scripts import configure_invokeai

        configure_invokeai.main()


if __name__ == "__main__":
    main()
