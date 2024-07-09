# Copyright (c) 2016-2024 Martin Donath <martin.donath@squidfunk.com>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

import glob
import json
import logging
import os
import platform
import regex
import requests
import site
import sys
import yaml

from colorama import Fore, Style
from importlib.metadata import distributions, version
from io import BytesIO
from markdown.extensions.toc import slugify
from mkdocs.config.defaults import MkDocsConfig
from mkdocs.plugins import BasePlugin, event_priority
from mkdocs.utils import get_yaml_loader
from zipfile import ZipFile, ZIP_DEFLATED

from .config import InfoConfig
from .patterns import get_exclusion_patterns

# -----------------------------------------------------------------------------
# Classes
# -----------------------------------------------------------------------------

# Info plugin
class InfoPlugin(BasePlugin[InfoConfig]):

    # Initialize plugin
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize incremental builds
        self.is_serve = False

        # Initialize empty members
        self.exclusion_patterns = []
        self.excluded_entries = []

    # Determine whether we're serving the site
    def on_startup(self, *, command, dirty):
        self.is_serve = command == "serve"

    # Create a self-contained example (run earliest) - determine all files that
    # are visible to MkDocs and are used to build the site, create an archive
    # that contains all of them, and print a summary of the archive contents.
    # The author must attach this archive to the bug report.
    @event_priority(100)
    def on_config(self, config):
        if not self.config.enabled:
            return

        # By default, the plugin is disabled when the documentation is served,
        # but not when it is built. This should nicely align with the expected
        # user experience when creating reproductions.
        if not self.config.enabled_on_serve and self.is_serve:
            return

        # Resolve latest version
        url = "https://github.com/squidfunk/mkdocs-material/releases/latest"
        res = requests.get(url, allow_redirects = False)

        # Check if we're running the latest version
        _, current = res.headers.get("location").rsplit("/", 1)
        present = version("mkdocs-material")
        if not present.startswith(current):
            log.error("Please upgrade to the latest version.")
            self._help_on_versions_and_exit(present, current)

        # Exit if archive creation is disabled
        if not self.config.archive:
            sys.exit(1)

        # Print message that we're creating a bug report
        log.info("Started archive creation for bug report")

        # Check that there are no overrides in place - we need to use a little
        # hack to detect whether the custom_dir setting was used without parsing
        # mkdocs.yml again - we check at which position the directory provided
        # by the theme resides, and if it's not the first one, abort.
        if config.theme.custom_dir:
            log.error("Please remove 'custom_dir' setting.")
            self._help_on_customizations_and_exit()

        # Check that there are no hooks in place - hooks can alter the behavior
        # of MkDocs in unpredictable ways, which is why they must be considered
        # being customizations. Thus, we can't offer support for debugging and
        # must abort here.
        if config.hooks:
            log.error("Please remove 'hooks' setting.")
            self._help_on_customizations_and_exit()

        # Assure all paths that will be validated are absolute. Convert possible
        # relative config_file_path to absolute. Its absolute directory path is
        # being later used to resolve other paths.
        config.config_file_path = _convert_to_abs(config.config_file_path)
        config_file_parent = os.path.dirname(config.config_file_path)

        # Convert relative custom_dir path to absolute. The Theme.custom_dir
        # property cannot be set, therefore a helper variable is used.
        if config.theme.custom_dir:
            abs_custom_dir = _convert_to_abs(
                config.theme.custom_dir,
                abs_prefix = config_file_parent
            )
        else:
            abs_custom_dir = ""

        # Extract the absolute path to projects plugin's directory to explicitly
        # support path validation and dynamic exclusion for the plugin
        projects_plugin = config.plugins.get("material/projects")
        if projects_plugin:
            abs_projects_dir = _convert_to_abs(
                projects_plugin.config.projects_dir,
                abs_prefix = config_file_parent
            )
        else:
            abs_projects_dir = ""

        # MkDocs removes the INHERIT configuration key during load, and doesn't
        # expose the information in any way, as the parent configuration is
        # merged into one. To validate that the INHERIT config file will be
        # included in the ZIP file the current config file must be loaded again
        # without parsing. Each file can have their own INHERIT key, so a list
        # of configurations is supported. The INHERIT path is converted during
        # load to absolute.
        loaded_configs = _load_yaml(config.config_file_path)
        if not isinstance(loaded_configs, list):
            loaded_configs = [loaded_configs]

        # We need to make sure the user put every file in the current working
        # directory. To assure the reproduction inside the ZIP file can be run,
        # validate that the MkDocs paths are children of the current root.
        paths_to_validate = [
            config.config_file_path,
            config.docs_dir,
            abs_custom_dir,
            abs_projects_dir,
            *[cfg.get("INHERIT", "") for cfg in loaded_configs]
        ]

        # Convert relative hook paths to absolute path
        for hook in config.hooks:
            path = _convert_to_abs(hook, abs_prefix = config_file_parent)
            paths_to_validate.append(path)

        # Remove valid paths from the list
        for path in list(paths_to_validate):
            if not path or path.startswith(os.getcwd()):
                paths_to_validate.remove(path)

        # Report the invalid paths to the user
        if paths_to_validate:
            log.error(f"One or more paths aren't children of root")
            self._help_on_not_in_cwd(paths_to_validate)

        # Create in-memory archive and prompt author for a short descriptive
        # name for the archive, which is also used as the directory name. Note
        # that the name is slugified for better readability and stripped of any
        # file extension that the author might have entered.
        archive = BytesIO()
        example = input("\nPlease name your bug report (2-4 words): ")
        example, _ = os.path.splitext(example)
        example = "-".join([present, slugify(example, "-")])

        # Get local copy of the exclusion patterns
        self.exclusion_patterns = get_exclusion_patterns()
        self.excluded_entries = []

        # Exclude the site_dir at project root
        if config.site_dir.startswith(os.getcwd()):
            self.exclusion_patterns.append(_resolve_pattern(config.site_dir))

        # Exclude the Virtual Environment directory. site.getsitepackages() has
        # inconsistent results across operating systems, and relies on the
        # PREFIXES that will contain the absolute path to the activated venv.
        for path in site.PREFIXES:
            if path.startswith(os.getcwd()):
                self.exclusion_patterns.append(_resolve_pattern(path))

        # Exclude site_dir for projects
        if projects_plugin:
            for path in glob.iglob(
                pathname = projects_plugin.config.projects_config_files,
                root_dir = abs_projects_dir,
                recursive = True
            ):
                current_config_file = os.path.join(abs_projects_dir, path)
                project_config = _get_project_config(current_config_file)
                pattern = _resolve_pattern(project_config.site_dir)
                self.exclusion_patterns.append(pattern)

        # Track dotpath inclusion to inform about it later
        contains_dotpath: bool = False

        # Create self-contained example from project
        files: list[str] = []
        with ZipFile(archive, "a", ZIP_DEFLATED, False) as f:
            for abs_root, dirnames, filenames in os.walk(os.getcwd()):
                # Set and print progress indicator
                indicator = f"Processing: {abs_root}"
                print(indicator, end="\r", flush=True)

                # Prune the folders in-place to prevent their processing
                for name in list(dirnames):
                    # Resolve the absolute directory path
                    path = os.path.join(abs_root, name)

                    # Exclude the directory and all subdirectories
                    if self._is_excluded(path):
                        dirnames.remove(name)
                        continue

                    # Warn about .dotdirectories
                    if _is_dotpath(path, log_warning = True):
                        contains_dotpath = True

                # Write files to the in-memory archive
                for name in filenames:
                    # Resolve the absolute file path
                    path = os.path.join(abs_root, name)

                    # Exclude the file
                    if self._is_excluded(path):
                        continue

                    # Warn about .dotfiles
                    if _is_dotpath(path, log_warning = True):
                        contains_dotpath = True

                    # Resolve the relative path to create a matching structure
                    path = os.path.relpath(path, os.path.curdir)
                    f.write(path, os.path.join(example, path))

                # Clear the line for the next indicator
                print(" " * len(indicator), end="\r", flush=True)

            # Add information on installed packages
            f.writestr(
                os.path.join(example, "requirements.lock.txt"),
                "\n".join(sorted([
                    "==".join([package.name, package.version])
                        for package in distributions()
                ]))
            )

            # Add information on platform
            f.writestr(
                os.path.join(example, "platform.json"),
                json.dumps(
                    {
                        "system": platform.platform(),
                        "architecture": platform.architecture(),
                        "python": platform.python_version(),
                        "cwd": os.getcwd(),
                        "command": " ".join([
                            sys.argv[0].rsplit(os.sep, 1)[-1],
                            *sys.argv[1:]
                        ]),
                        "env:$PYTHONPATH": os.getenv("PYTHONPATH", ""),
                        "sys.path": sys.path,
                        "excluded_entries": self.excluded_entries
                    },
                    default = str,
                    indent = 2
                )
            )

            # Retrieve list of processed files
            for a in f.filelist:
                # Highlight .dotpaths in a more explicit manner
                color = (Fore.LIGHTYELLOW_EX if "/." in a.filename
                         else Fore.LIGHTBLACK_EX)
                files.append("".join([
                    color, a.filename, " ",
                    _size(a.compress_size)
                ]))

        # Finally, write archive to disk
        buffer = archive.getbuffer()
        with open(f"{example}.zip", "wb") as f:
            f.write(archive.getvalue())

        # Print summary
        log.info("Archive successfully created:")
        print(Style.NORMAL)

        # Print archive file names
        files.sort()
        for file in files:
            print(f"  {file}")

        # Print archive name
        print(Style.RESET_ALL)
        print("".join([
            "  ", f.name, " ",
            _size(buffer.nbytes, 10)
        ]))

        # Print warning when file size is excessively large
        print(Style.RESET_ALL)
        if buffer.nbytes > 1000000:
            log.warning("Archive exceeds recommended maximum size of 1 MB")

        # Print warning when file contains hidden .dotpaths
        if contains_dotpath:
            log.warning(
                "Archive contains dotpaths, which could contain sensitive "
                "information.\nPlease review them at the bottom of the list "
                "and share only necessary data to reproduce the issue."
            )

        # Aaaaaand done
        sys.exit(1)

    # -------------------------------------------------------------------------

    # Print help on versions and exit
    def _help_on_versions_and_exit(self, have, need):
        print(Fore.RED)
        print("  When reporting issues, please first upgrade to the latest")
        print("  version of Material for MkDocs, as the problem might already")
        print("  be fixed in the latest version. This helps reduce duplicate")
        print("  efforts and saves us maintainers time.")
        print(Style.NORMAL)
        print(f"  Please update from {have} to {need}.")
        print(Style.RESET_ALL)
        print(f"  pip install --upgrade --force-reinstall mkdocs-material")
        print(Style.NORMAL)

        # Exit, unless explicitly told not to
        if self.config.archive_stop_on_violation:
            sys.exit(1)

    # Print help on customizations and exit
    def _help_on_customizations_and_exit(self):
        print(Fore.RED)
        print("  When reporting issues, you must remove all customizations")
        print("  and check if the problem persists. If not, the problem is")
        print("  caused by your overrides. Please understand that we can't")
        print("  help you debug your customizations. Please remove:")
        print(Style.NORMAL)
        print("  - theme.custom_dir")
        print("  - hooks")
        print(Fore.YELLOW)
        print("  Additionally, please remove all third-party JavaScript or")
        print("  CSS not explicitly mentioned in our documentation:")
        print(Style.NORMAL)
        print("  - extra_css")
        print("  - extra_javascript")
        print(Fore.YELLOW)
        print("  If you're using customizations from the theme's documentation")
        print("  and you want to report a bug specific to those customizations")
        print("  then set the 'archive_stop_on_violation: false' option in the")
        print("  info plugin config.")
        print(Style.RESET_ALL)

        # Exit, unless explicitly told not to
        if self.config.archive_stop_on_violation:
            sys.exit(1)

    # Print help on not in current working directory and exit
    def _help_on_not_in_cwd(self, outside_root):
        print(Fore.RED)
        print("  The current working (root) directory:\n")
        print(f"    {os.getcwd()}\n")
        print("  is not a parent of the following paths:")
        print(Style.NORMAL)
        for path in outside_root:
            print(f"    {path}")
        print("\n  To assure that all project files are found please adjust")
        print("  your config or file structure and put everything within the")
        print("  root directory of the project.")
        print("\n  Please also make sure `mkdocs build` is run in the actual")
        print("  root directory of the project.")
        print(Style.RESET_ALL)

        # Exit, unless explicitly told not to
        if self.config.archive_stop_on_violation:
            sys.exit(1)

    # Check if path is excluded and should be omitted from the zip. Use pattern
    # matching for files and folders, and lookahead specific files in folders to
    # skip them. Side effect: Save excluded paths to save them in the zip file.
    def _is_excluded(self, abspath: str) -> bool:

        # Resolve the path into POSIX format to match the patterns
        pattern_path = _resolve_pattern(abspath, return_path = True)

        for pattern in self.exclusion_patterns:
            if regex.search(pattern, pattern_path):
                log.debug(f"Excluded pattern '{pattern}': {abspath}")
                self.excluded_entries.append(f"{pattern} - {pattern_path}")
                return True

        # File exclusion should be limited to pattern matching
        if os.path.isfile(abspath):
            return False

        # Projects, which don't use the projects plugin for multi-language
        # support could have separate build folders for each config file or
        # language. Therefore, we exclude them with the assumption a site_dir
        # contains the sitemap file. Example of such a setup: https://t.ly/DLQcy
        sitemap_gz = os.path.join(abspath, "sitemap.xml.gz")
        if os.path.exists(sitemap_gz):
            log.debug(f"Excluded site_dir: {abspath}")
            self.excluded_entries.append(f"sitemap.xml.gz - {pattern_path}")
            return True

        return False

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

# Print human-readable size
def _size(value, factor = 1):
    color = Fore.GREEN
    if   value > 100000 * factor: color = Fore.RED
    elif value >  25000 * factor: color = Fore.YELLOW
    for unit in ["B", "kB", "MB", "GB", "TB", "PB", "EB", "ZB"]:
        if abs(value) < 1000.0:
            return f"{color}{value:3.1f} {unit}"
        value /= 1000.0

# Get the absolute path with set prefix. To validate if a file is inside the
# current working directory it needs to be absolute, so that it is possible to
# check the prefix.
def _convert_to_abs(path: str, abs_prefix: str = None) -> str:
    if os.path.isabs(path): return path
    if abs_prefix is None: abs_prefix = os.getcwd()
    return os.path.normpath(os.path.join(abs_prefix, path))

# Get the loaded config, or a list with all loaded configs. MkDocs removes the
# INHERIT configuration key during load, and doesn't expose the information in
# any way, as the parent configuration is merged into one. The INHERIT path is
# needed for validation. This custom YAML loader replicates MkDocs' loading
# logic. Side effect: It converts the INHERIT path to absolute.
def _load_yaml(abs_src_path: str):

    with open(abs_src_path, encoding ="utf-8-sig") as file:
        source = file.read()

    try:
        result = yaml.load(source, Loader = get_yaml_loader()) or {}
    except yaml.YAMLError:
        result = {}

    if "INHERIT" in result:
        relpath = result.get('INHERIT')
        parent_path = os.path.dirname(abs_src_path)
        abspath = _convert_to_abs(relpath, abs_prefix = parent_path)
        if os.path.exists(abspath):
            result["INHERIT"] = abspath
            log.debug(f"Loading inherited configuration file: {abspath}")
            parent = _load_yaml(abspath)
            if isinstance(parent, list):
                result = [result, *parent]
            elif isinstance(parent, dict):
                result = [result, parent]

    return result

# Get a normalized POSIX path for the pattern matching with removed current
# working directory prefix. Directory paths end with a '/' to allow more control
# in the pattern creation for files and directories. The patterns are matched
# using the search function, so they are prefixed with ^ for specificity.
def _resolve_pattern(abspath: str, return_path: bool = False):
    path = abspath.replace(os.getcwd(), "", 1)
    path = path.replace(os.sep, "/").rstrip("/")

    if not path:
        return "/"

    # Check abspath, as the file needs to exist
    if not os.path.isfile(abspath):
        path = path + "/"

    return path if return_path else f"^{path}"

# Get project configuration with resolved absolute paths for validation
def _get_project_config(project_config_file: str):
    with open(project_config_file, encoding="utf-8-sig") as file:
        config = MkDocsConfig(config_file_path = project_config_file)
        config.load_file(file)

        # MkDocs transforms site_dir to absolute path during validation
        config.validate()

        return config

# Check if the path is a .dotpath. A warning can also be issued when the param
# is set. The function also returns a boolean to track results outside it.
def _is_dotpath(path: str, log_warning: bool = False) -> bool:
    posix_path = _resolve_pattern(path, return_path = True)
    name = posix_path.rstrip("/").rsplit("/", 1)[-1]
    if name.startswith("."):
        if log_warning:
            log.warning(f"The following .dotpath will be included: {path}")
        return True
    return False


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

# Set up logging
log = logging.getLogger("mkdocs.material.info")
