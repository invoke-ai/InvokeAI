#!/usr/bin/env python3
"""
script to setup the invokeai root directory
destination is the current working directory, unless Globals.root is set
this script will not overwrite any existing directorys
"""
from os import getcwd, getenv, listdir, path
from shutil import copytree
from sys import platlibdir, prefix

import configure_invokeai

dirs_to_copy = ["configs", "frontend", "scripts"]


def copy_from_venv(folders) -> None:
    """
    function to copy directorys from the venv's tree from source_prefix to Globals.root
    usefull since the config script will not work in a empty directory.
    """
    done = int(0)

    for folder in folders:
        if getenv("INVOKEAI_ROOT"):
            dest_folder = path.join(getenv("INVOKEAI_ROOT"), folder)
        else:
            dest_folder = path.join(getcwd(), folder)

        source_prefix = path.join(
            prefix,
            platlibdir,
            listdir(path.join(prefix, platlibdir))[0],
            "site-packages",
        )
        src_folder = path.abspath(path.join(source_prefix, folder))

        try:
            # print(f"trying to copy {src_folder} to {dest_folder}")
            copytree(src_folder, dest_folder)
            done += 1
        except FileExistsError as error:
            print(error)
            done += 1
        except FileNotFoundError as error:
            print(error)
        except RuntimeError as error:
            print(error)

        # if succeesfull, run configure_invokeai afterwards◊
        if len(folders) == done:
            configure_invokeai.main()


if __name__ == "__main__":
    copy_from_venv(folders=dirs_to_copy)
