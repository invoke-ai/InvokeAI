# Steps for making a release

## Initial Steps

1. Make a developer call-out for PRs to merge. Merge and test.
1. Run "make ruff"
1. Run "pytest tests"

## Prepare release branch

1. Create a release branch and create a pull request for it against `main`.
   - This is not strictly necessary, but I have found that during the pre-release testing process
     multiple small bugs are uncovered. Having an unprotected branch available to push changes to
     allows you to iterate more rapidly. Otherwise all individual changes will need to go through
     the PR review and testing process.
   - **New** If the branch starts with `release/`, as in `release/4.0.0`, then you can release the
     branch directly and it will be uploaded to PyPi. Otherwise, the branch must be merged into
     `main` and the release performed from a tag on that branch.
2. Edit the file `invokeai/version/invokeai_version.py` to bring the InvokeAI version up to the
   desired release number. I refer to <https://peps.python.org/pep-0440/> for the accepted formats
3. Deactivate any active virtual environment (the next step will complain if you don't)
4. Run `make installer-zip`. This will build the frontend, build a wheel of the entire InvokeAI
   distribution, and then package the wheel into a .zip file with the installer script.
5. Test the zip file
   - Unzip it to a temporary directory
   - Run its `install.sh` script
   - Do basic checking that root directory is created, starter models are installed, and then web
     server starts up and can generate images.

## Upload Release (candidate) to GitHub

3. When satisfied with the installer, tag this commit with `make tag-release`. Note that if there is
   already a tag with the same version number, this script will delete the old tag and create a new
   one on the current commit. These operations directly affect the GitHub repository, so be careful!
4. Go to the [InvokeAI Releases Page](https://github.com/invoke-ai/InvokeAI/releases) and push the
   _Draft New Release_ button.
5. You will be prompted to select or create a tag for the release. Select the one chosen in step
   (2).
6. Write the release notes. There is a GitHub button labeled _Generate release notes_ that will
   automatically insert the changelog and new contributors. I use this to generate the bottom half
   of the release notes, and the tried and true method of cutting and pasting from the previous
   release notes to write the intro.
7. Upload the zip file created in step 4 into the Assets section of the release notes. I also like
   to upload them into the body of the release notes, since it can be hard for users to find the
   Assets section.
8. Check the _Set as a pre-release_ and \_Create discussion item` checkboxes at the bottom of the
   release page, and Save.
9. Announce the pre-release in Discussion and Discord.

## Revisions

When a change has been made to the release branch, bump up the version number in `_version.py` and
repeat steps 4-9. Generally you only need to upload the new versions of the zip file and let the
users know that a change has been made. Use the popup menu on the upper left of the release page to
update the pre-release to the new version-patchlevel tag.

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

1. On the `main` branch, bump up the version number to X.Y.Za1 to indicate that this is now an alpha
   release. If you don't do this, then people who try to update to `main` will not get the most
   recent version because of pip's version checking logic.
