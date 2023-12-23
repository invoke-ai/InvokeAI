# Release Workflow

The app is published in twice, in different build formats.

- A [PyPI] distribution. This includes both a source distribution and built distribution (a wheel). Users install with `pip install invokeai`. The updater uses this build.
- An installer on the [InvokeAI Releases Page]. This is a zip file with install scripts and a wheel. This is only used for new installs.

## General Prep

Make a developer call-out for PRs to merge. Merge and test things out.

While the release workflow does not include end-to-end tests, it does pause before publishing so you can download and test the final build.

## Workflow Overview

The `release.yml` workflow runs a number of jobs to handle code checks, tests, build and publish on PyPI.

It is triggered on **tag push**, when the tag matches `v*.*.*`. It doesn't matter if you've prepped a release branch like `release/v3.5.0` or are releasing from `main` - it works the same.

!!! tip

    Because commits are reference-counted, it is safe to create a release branch, tag it, let the workflow run, then delete the branch.

    So long as the tag exists, that commit will exist.

### Triggering the Workflow

Run `make tag-release` to tag the current commit and kick off the workflow.

This script actually makes two tags - one for the specific version, and a `vX-latest` tag that changes with each release.

Because the release workflow only triggers on the pattern `v*.*.*`, the workflow will only run once when running this script.

The release may also be run [manually].

### Workflow Jobs and Process

The workflow consists of a number of concurrently-run jobs, and two final publish jobs.

The publish jobs run if the 5 concurrent jobs all succeed and if/when the publish jobs are approved.

#### `check-version` Job

This job checks that the git ref matches the app version. It matches the ref against the `__version__` variable in `invokeai/version/invokeai_version.py`.

When the workflow is triggered by tag push, the ref is the tag. If the workflow is run manually, the ref is the target selected from the **Use workflow from** dropdown.

!!! tip

    Any valid [version specifier] works, so long as the tag matches the version. The release workflow works exactly the same for `RC`, `post`, `dev`, etc.

#### Check and Test Jobs

This is our test suite.

- **`check-pytest`**: runs `pytest` on matrix of platforms
- **`check-python`**: runs `ruff` (format and lint)
- **`check-frontend`**: runs `prettier` (format), `eslint` (lint), `madge` (circular refs) and `tsc` (static type check)

!!! info Future Enhancement

    We should add `mypy` or `pyright` to the **`check-python`** job.

!!! info Future Enhancement

    We should add an end-to-end test job that generates an image.

#### `build` Job

This sets up both python and frontend dependencies and builds the python package. Internally, this runs `installer/create_installer.sh` and uploads two artifacts:

- **`dist`**: the python distribution, to be published on PyPI
- **`InvokeAI-installer-${VERSION}.zip`**: the installer to be included in the GitHub release

#### Sanity Check & Smoke Test

At this point, the release workflow pauses (the remaining jobs all require approval).

A maintainer should go to the **Summary** tab of the workflow, download the installer and test it. Ensure the app loads and generates.

!!! info

    The exact same wheel file is bundled in the installer and in the `dist` artifact, which is uploaded to PyPI. You should end up with the same exact installation of the `invokeai` package from any of these methods.

#### PyPI Publish Jobs

The publish jobs will skip if any of the previous jobs skip or fail.

They use [GitHub environments], which are configured as [trusted publishers] on PyPI.

Both jobs require a maintainer to approve them from the workflow's **Summary** tab.

- Click the **Review deployments** button
- Select the environment (either `testpypi` or `pypi`)
- Click **Approve and deploy**

!!! warning

    **If the version already exists on PyPI, the publish jobs will fail.** PyPI only allows a particular version to be published once - you cannot change it. If version published on PyPI has a problem, you'll need to "fail forward" by bumping the app version and publishing a followup release.

#### `publish-testpypi` Job

Publishes the distribution on the [Test PyPI] index, using the `testpypi` GitHub environment.

This job is not required for the production PyPI publish, but included just in case you want to test the PyPI release.

If approved and successful, you could try out the test release like this:

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

#### `publish-pypi` Job

Publishes the distribution on the production PyPI index, using the `pypi` GitHub environment.

## Publish the GitHub RC with installer

1. [Draft a new release] on GitHub, choosing the tag that triggered the release.
2. Write the release notes, describing important changes. The **Generate release notes** button automatically inserts the changelog and new contributors, and you can copy/paste the intro from previous releases.
3. Upload the zip file created in [Build the installer] into the Assets section of the release notes. You can also upload the zip into the body of the release notes, since it can be hard for users to find the Assets section.
4. Check the **Set as a pre-release** and **Create a discussion for this release** checkboxes at the bottom of the release page.
5. Publish the pre-release.
6. Announce the pre-release in Discord.

!!! info Future Enhancement

    Workflows can create a GitHub release from a template and upload release assets. One popular action to handle this is [ncipollo/release-action]. A future enhancement to the release process could set this up.

## Manually Running the Release Workflow

The release workflow can be run manually. This is useful to get an installer build and test it out without needing to push a tag.

When run this way, you'll see **Skip code checks** checkbox. This allows the workflow to run without the time-consuming 3 code quality check jobs.

The publish jobs will skip if the workflow was run manually.

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
