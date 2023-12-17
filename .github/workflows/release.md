# Releasing InvokeAI

Releases are published in twice, in different build formats.

- A [PyPI] distribution. This includes both a source distribution and built distribution (a wheel). Users install with `pip install invokeai`. The updater uses this build.
- An installer on the [InvokeAI Releases Page]. This is a zip file with install scripts and a wheel. This is only used for new installs.

## General Prep

1. Make a developer call-out for PRs to merge. Merge and test.
1. Run "make ruff", linting and formatting the codebase.
1. Run "pytest tests", ensuring the test suite passes with no failures.

## Create a release branch

1. Choose the new version. It should adhere to the PyPA [version specifiers] spec.
1. Create a branch called `release/$VERSION`.
1. Edit `invokeai/version/invokeai_version.py`, bumping the version.
1. Create a PR with this branch.

## Build the installer

1. Deactivate any active virtual environment.
1. Run `make installer-zip`. This will build the frontend, build a wheel of the entire InvokeAI distribution, and then package the wheel into a .zip file with the installer script.

## Test the installer

1. Unzip the installer to a temporary directory.
1. Run the `install.sh` script.
1. Confirm that the root directory is created, starter models are installed, and the web server starts up and can generate images.

## Publish on PyPI

1. Navigate to the [PyPI release workflow] on GitHub, click `Run Workflow` and select the `release/$VERSION` branch created earlier.
1. Enter `true` in the `Publish build on PyPi? [true/false]` prompt.
1. Run the workflow. It will take 2 or 3 minutes.

!!! info Dry run

    To do a dry run of the PyPI build, enter `false` at the prompt when running the workflow. This will build the distribution but not publish it to PyPI. You can then download this as distribution from the **Summary** tab and install the wheel locally to test.

    You can run the workflow on any branch as a dry run, but it will only publish from branches named `release/<something>` or from `main`.

## Tag the release

From the repo root, on your `release/$VERSION` branch, run `make tag-release`.

This script does two things:

- Creates tag `v$VERSION` pointing to the current commit, e.g. `v3.4.0`
- Creates tag `v$VERSION_MAJOR-latest` pointing to the current commit, e.g. `v3-latest`

!!! warning

    If there's already a tag with the same version number, this script will delete the old tag and create a new one pointing to the current commit. These operations directly affect the GitHub repo, so be careful!

## Publish the GitHub RC with installer

1. [Draft a new release] on GitHub, choosing the tag created in [Tag the release].
1. Write the release notes, describing important changes. The **Generate release notes** button automatically inserts the changelog and new contributors, and you can copy/paste the intro from previous releases.
1. Upload the zip file created in [Build the installer] into the Assets section of the release notes. You can also upload the zip into the body of the release notes, since it can be hard for users to find the Assets section.
1. Check the **Set as a pre-release** and **Create a discussion for this release** checkboxes at the bottom of the release page.
1. Publish the pre-release.
1. Announce the pre-release in Discord.

## Merge the release branch

At this point, the

## Revisions

When a change has been made to the release branch, bump up the version number in `_version.py` and repeat steps 4-9. Generally you only need to upload the new versions of the zip file and let the users know that a change has been made. Use the popup menu on the upper left of the release page to update the pre-release to the new version-patchlevel tag.

Iterate this process until you are ready to finalize the release.

## Final Release from the Release Branch

If you named the branch `release/<something>` then you can release it directly:

1. Update the version number to what you want to see in the release.
2. Build the zip file as before
3. Tag the commit as before
4. Upload the release zip file to the release page.
5. Update the tag to use in the release page.
6. Change the "pre-release" checkbox to "latest release"
7. Push the green button.

## Final Release from the Main Branch

Alternatively you can release from `main` (this was the traditional way of doing it).

Approve the release branch's PR and merge into `main`. Now, working in the `main` branch:

1. Run `create_installer.sh` to do the final tagging and zip file generation.
2. Merge to `main` (important!). This will move the tag to main as well.
3. Open the pre-release page in edit mode and upload the final zip files.
4. Choose the final tag from the popup menu.
5. Uncheck `Set as pre-release` and check `Set as the latest release`.
6. Save, and the release is done!

## Publicity

1. Create a Discussion announcement
2. Announce on Discord and Reddit
3. Produce and release a YouTube video, if appropriate.

## Post Release Step (new!)

1. On the `main` branch, bump up the version number to X.Y.Za1 to indicate that this is now an alpha release. If you don't do this, then people who try to update to `main` will not get the most recent version because of pip's version checking logic.

[version specifiers]: https://packaging.python.org/en/latest/specifications/version-specifiers/
[InvokeAI Releases Page]: https://github.com/invoke-ai/InvokeAI/releases
[PyPI]: https://pypi.org/
[PyPI release workflow]: https://github.com/invoke-ai/InvokeAI/actions/workflows/pypi-release.yml
[Tag the release]: #tag-the-release
[Build the installer]: #build-the-installer
[Draft a new release]: https://github.com/invoke-ai/InvokeAI/releases/new
