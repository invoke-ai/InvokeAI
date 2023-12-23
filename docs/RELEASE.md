# Release

The app is published in twice, in different build formats.

- A [PyPI] distribution. This includes both a source distribution and built distribution (a wheel). Users install with `pip install invokeai`. The updater uses this build.
- An installer on the [InvokeAI Releases Page]. This is a zip file with install scripts and a wheel. This is only used for new installs.

## General Prep

Make a developer call-out for PRs to merge. Merge and test things out.

While the release workflow does not include end-to-end tests, it does pause before publishing so you can download and test the final build.

## Release Workflow

The `release.yml` workflow runs a number of jobs to handle code checks, tests, build and publish on PyPI.

It is triggered on **tag push**. It doesn't matter if you've prepped a release branch like `release/v3.5.0` or are releasing from `main` - it works the same.

!!! info

    Commits are reference-counted, so as long as a something points to a commit, that commit will not be garbage-collected'd from the repo.

    It is safe to create a release branch, tag it and have the workflow do its thing, then delete the branch. So long as the tag is not deleted, that snapshot of the repo will forever exist at the tag.

### Tag Push Example

Any tag push will trigger the workflow, but it will publish only if the git ref (the tag) matches the app version.

Say `invokeai_version.py` looks like this:

```py
__version__ = "3.5.0rc2"
```

- If you push tag `v3.5.0rc2`, the workflow will trigger and run. If the checks and build succeed, you'll be able to publish the release.

- If you push tag `v3.5.0rc3` or `banana-sushi`, the workflow will trigger and run. Even if the checks and build succeed, you'll _will not_ be able to publish the release, because the tag doesn't match the app version.

!!! info

    Any valid [version specifier] works, so long as the tag matches the version. The release workflow works exactly the same for `RC`, `post`, `dev`, etc.

### code quality jobs

Three jobs are run concurrently:

- **`pytest`**: runs `pytest` on matrix of platforms
- **`check-python`**: runs `ruff` (format and lint)
- **`check-frontend`**: runs `prettier` (format), `eslint` (lint), `madge` (circular refs) and `tsc` (static type check)

If any fail, the release workflow bails.

!!! info Future Enhancement

    We should add `mypy` or `pyright` to the **`check-python`** job at some point.

### `build`

This sets up both python and frontend dependencies and builds the python package. Internally, this runs `installer/create_installer.sh` and uploads two artifacts:

- **`dist`**: the python distribution, to be published on PyPI
- **`InvokeAI-installer-${VERSION}.zip`**: the installer to be included in the GitHub release

!!! info

    The installer uses the uses the same wheel file that is included in the PyPI distribution, so you _should_ get exactly the same thing using the installer or PyPI dist.

### Sanity Check & Smoke Test

At this point, the release workflow pauses (the remaining jobs all require approval).

The maintainer who is running this release should go to the **Summary** tab of the workflow, download the installer and test it.

You could also download the `dist`, unzip it and install directly from the wheel. That same wheel will be uploaded to PyPI.

### Publish

The publish jobs use [GitHub environments], which are configured as [trusted publishers] on PyPI.

Both jobs require a maintainer to approve them from the workflow's **Summary** tab.

- Click the **Review deployments** button
- Select the environment (either `testpypi` or `pypi`)
- Click **Approve and deploy**

#### Skip and Failure Conditions

The publish jobs may skip or fail in certain situations:

- **If code checks or build fail, the jobs will be skipped.**
- **If code checks were skipped, the jobs will be skipped.** This can only happen when [manually] running the workflow.
- **If the git ref targetted by the workflow doesn't match the app version, the jobs will fail.** This protects us from accidentally publishing the wrong version to PyPI. This is achieved with [samuelcolvin/check-python-version].
- **If the version already exists on PyPI, the jobs will fail.** PyPI only allows a particular version to be published once - you cannot change it. If version published on PyPI has a problem, you'll need to "fail forward" by bumping the app version and publishing a followup release.

#### `publish-testpypi`

Publishes the distribution on the [Test PyPI] index using the `testpypi` GitHub environment.

This job is optional:

- It is not require for the final `publish-pypi` job to run.
- The wheel used in the installer and PyPI dist (uploaded as artifacts from the workflow, as described above) are identical, so this job _should_ be extraneous.

That said, you could approve it and then test installing from PyPI before running the production PyPI publish job:

```sh
# Create a new virtual environment
python -m venv ~/.test-invokeai-dist --prompt test-invokeai-dist
# Install the distribution from Test PyPI
pip install --index-url https://test.pypi.org/simple/ invokeai
# Run and test the app
invokeai-web
# Cleanup
deactivate
rm -rf ~/.test-invokeai-dist
```

#### `publish-pypi`

Publishes the distribution on the production PyPI index, using the `pypi` GitHub environment.

Once this finishes, `pip install invokeai` will get the release!

## Publish the GitHub RC with installer

1. [Draft a new release] on GitHub, choosing the tag that initiated the release.
2. Write the release notes, describing important changes. The **Generate release notes** button automatically inserts the changelog and new contributors, and you can copy/paste the intro from previous releases.
3. Upload the zip file created in [Build the installer] into the Assets section of the release notes. You can also upload the zip into the body of the release notes, since it can be hard for users to find the Assets section.
4. Check the **Set as a pre-release** and **Create a discussion for this release** checkboxes at the bottom of the release page.
5. Publish the pre-release.
6. Announce the pre-release in Discord.

!!! info Future Enhancement

    Workflows can do things like create a release from a template and upload release assets. One popular action to handle this is [ncipollo/release-action]. A future enhancement to the release process could set this up.

## Manually Running the Release Workflow

The release workflow can be kicked off manually. This is useful to get an installer build and test it out without needing to push a tag.

When run this way, you'll see **Skip code checks** checkbox. This allows the workflow to run without the time-consuming 3 code quality check jobs. The publish jobs will be skipped if enabled.

[InvokeAI Releases Page]: https://github.com/invoke-ai/InvokeAI/releases
[PyPI]: https://pypi.org/
[Draft a new release]: https://github.com/invoke-ai/InvokeAI/releases/new
[Test PyPI]: https://test.pypi.org/
[version specifier]: https://packaging.python.org/en/latest/specifications/version-specifiers/
[ncipollo/release-action]: https://github.com/ncipollo/release-action
[GitHub environments]: https://docs.github.com/en/actions/deployment/targeting-different-environments/using-environments-for-deployment
[trusted publishers]: https://docs.pypi.org/trusted-publishers/
[samuelcolvin/check-python-version]: https://github.com/samuelcolvin/check-python-version
[manually]: #manually-running-the-release-workflow
