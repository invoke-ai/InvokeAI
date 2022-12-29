"""
script to create the necesarry folder structure
"""
import shutil
import os
# from ldm.invoke.globals import Globals

def main():
    """
    main function
    """
    osp = os.path
    dirs_to_copy = 'configs', 'frontend', 'scripts'
    done = int()

    def copy_from_venv(folder):
        """
        function to copy the requested directory tree from the venv
        """
        src_folder = osp.abspath(osp.join(os.environ.get('VIRTUAL_ENV'), folder))
        dest_folder = osp.abspath(osp.join(os.environ.get('VIRTUAL_ENV'), '..', folder))
        # dest_folder = osp.join(Globals.root, folder)
        print(f'trying to copy {src_folder} to {dest_folder}')
        shutil.copytree(src_folder, dest_folder)

    for tree in dirs_to_copy:
        try:
            copy_from_venv(folder = tree)
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
