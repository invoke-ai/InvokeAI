"""
script to setup the current working directory as the invokeai_root
(unless Globals.root is set) will not overwrite existing directorys
"""
import os
import shutil
import sys


def main():
    """
    main function
    """
    osp = os.path

    def copy_from_venv(folder):
        """
        function to copy directorys from the venv's  tree from source_prefix to Globals.root
        usefull since the config script will not work in a empty directory.
        """
        if os.getenv("INVOKEAI_ROOT"):
            dest_folder = osp.abspath(osp.join(os.getenv("INVOKEAI_ROOT"), folder))
        else:
            dest_folder = osp.abspath(osp.join(os.getcwd(), folder))

        source_prefix = os.path.join(
            sys.prefix,
            sys.platlibdir,
            os.listdir(os.path.join(sys.prefix, sys.platlibdir))[0],
            "site-packages",
        )
        src_folder = osp.abspath(osp.join(source_prefix, folder))
        print(f"trying to copy {src_folder} to {dest_folder}")
        shutil.copytree(src_folder, dest_folder)

    # run the copy job
    dirs_to_copy = "configs", "frontend", "scripts"
    done = 0
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

    # if succeesfull, run configure_invokeai afterwards◊
    if int(len(dirs_to_copy)) == done:
        from scripts import configure_invokeai

        configure_invokeai.main()


if __name__ == "__main__":
    main()
