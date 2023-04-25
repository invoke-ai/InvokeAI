'''ldm.invoke.config.post_install

This defines a single exportable function, post_install(), which does
post-installation stuff like migrating models directories, adding new
config files, etc.

From the command line, its entry point is invokeai-postinstall.
'''

import os
import sys
from packaging import version
from pathlib import Path
from shutil import move,rmtree,copyfile
from typing import Union

import invokeai.configs as conf
import ldm.invoke
from ..globals import Globals, global_cache_dir, global_config_dir

def post_install():
    '''
    Do version and model updates, etc.
    Should be called once after every version update.
    '''
    _migrate_models()
    _run_patches()


def _migrate_models():
    """
    Migrate the ~/invokeai/models directory from the legacy format used through 2.2.5
    to the 2.3.0 "diffusers" version. This should be a one-time operation, called at
    script startup time.
    """
    # Three transformer models to check: bert, clip and safety checker, and
    # the diffusers as well
    models_dir = Path(Globals.root, "models")
    legacy_locations = [
        Path(
            models_dir,
            "CompVis/stable-diffusion-safety-checker/models--CompVis--stable-diffusion-safety-checker",
        ),
        Path("bert-base-uncased/models--bert-base-uncased"),
        Path(
            "openai/clip-vit-large-patch14/models--openai--clip-vit-large-patch14"
        ),
    ]
    legacy_locations.extend(list(global_cache_dir("diffusers").glob("*")))
    legacy_layout = False
    for model in legacy_locations:
        legacy_layout = legacy_layout or model.exists()
    if not legacy_layout:
        return

    print(
        """
>> ALERT:
>> The location of your previously-installed diffusers models needs to move from
>> invokeai/models/diffusers to invokeai/models/hub due to a change introduced by
>> diffusers version 0.14. InvokeAI will now move all models from the "diffusers" directory
>> into "hub" and then remove the diffusers directory. This is a quick, safe, one-time
>> operation. However if you have customized either of these directories and need to
>> make adjustments, please press ctrl-C now to abort and relaunch InvokeAI when you are ready.
>> Otherwise press <enter> to continue."""
    )
    print("** This is a quick one-time operation.")
    input("continue> ")

    # transformer files get moved into the hub directory
    if _is_huggingface_hub_directory_present():
        hub = global_cache_dir("hub")
    else:
        hub = models_dir / "hub"

    os.makedirs(hub, exist_ok=True)
    for model in legacy_locations:
        source = models_dir / model
        dest = hub / model.stem
        if dest.exists() and not source.exists():
            continue
        print(f"** {source} => {dest}")
        if source.exists():
            if dest.is_symlink():
                print(f"** Found symlink at {dest.name}. Not migrating.")
            elif dest.exists():
                if source.is_dir():
                    rmtree(source)
                else:
                    source.unlink()
            else:
                move(source, dest)

    # now clean up by removing any empty directories
    empty = [
        root
        for root, dirs, files, in os.walk(models_dir)
        if not len(dirs) and not len(files)
    ]
    for d in empty:
        os.rmdir(d)
    print("** Migration is done. Continuing...")


def _is_huggingface_hub_directory_present() -> bool:
    return (
        os.getenv("HF_HOME") is not None or os.getenv("XDG_CACHE_HOME") is not None
    )

# This routine performs any patch-ups needed after installation
def _run_patches():
    _install_missing_config_files()
    version_file = Path(Globals.root,'.version')
    if version_file.exists():
        with open(version_file,'r') as f:
            root_version = version.parse(f.readline() or 'v2.3.2')
    else:
        root_version = version.parse('v2.3.2')
    app_version = version.parse(ldm.invoke.__version__)
    if root_version < app_version:
        try:
            _do_version_update(root_version, ldm.invoke.__version__)
            with open(version_file,'w') as f:
                f.write(ldm.invoke.__version__)
        except:
            print("** Version patching failed. Please try invokeai-postinstall later.")

def _install_missing_config_files():
    """
    install ckpt configuration files that may have been added to the
    distro after original root directory configuration
    """
    root_configs = Path(global_config_dir(), 'stable-diffusion')
    repo_configs = None
    for f in conf.__path__:
        if Path(f, 'stable-diffusion', 'v1-inference.yaml').exists():
            repo_configs = Path(f, 'stable-diffusion')
            break
    if not repo_configs:
        return
    for src in repo_configs.iterdir():
        dest = root_configs / src.name
        if not dest.exists():
            copyfile(src,dest)
    
def _do_version_update(root_version: version.Version, app_version: Union[str, version.Version]):
    """
    Make any updates to the launcher .sh and .bat scripts that may be needed
    from release to release. This is not an elegant solution. Instead, the 
    launcher should be moved into the source tree and installed using pip.
    """
    if root_version < version.Version('v2.3.4'):
        dest = Path(Globals.root,'loras')
        dest.mkdir(exist_ok=True)
    if root_version < version.Version('v2.3.3'):
        if sys.platform == "linux":
            print('>> Downloading new version of launcher script and its config file')
            from ldm.util import download_with_progress_bar
            url_base = f'https://raw.githubusercontent.com/invoke-ai/InvokeAI/v{str(app_version)}/installer/templates/'

            dest = Path(Globals.root,'invoke.sh.in')
            assert download_with_progress_bar(url_base+'invoke.sh.in',dest)
            dest.replace(Path(Globals.root,'invoke.sh'))
            os.chmod(Path(Globals.root,'invoke.sh'), 0o0755)

            dest = Path(Globals.root,'dialogrc')
            assert download_with_progress_bar(url_base+'dialogrc',dest)
            dest.replace(Path(Globals.root,'.dialogrc'))
