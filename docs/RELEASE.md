# Release Process

The app is published in twice, in different build formats.

- A [PyPI] distribution. This includes both a source distribution and built distribution (a wheel). Users install with `pip install invokeai`. The updater uses this build.
- An installer on the [InvokeAI Releases Page]. This is a zip file with install scripts and a wheel. This is only used for new installs.

## General Prep

Make a developer call-out for PRs to merge. Merge and test things out.

While the release workflow does not include end-to-end tests, it does pause before publishing so you can download and test the final build.

## Release Workflow

The `release.yml` workflow runs a number of jobs to handle code checks, tests, build and publish on PyPI.

It is triggered on **tag push**, when the tag matches `v*`. It doesn't matter if you've prepped a release branch like `release/v3.5.0` or are releasing from `main` - it works the same.

> Because commits are reference-counted, it is safe to create a release branch, tag it, let the workflow run, then delete the branch. So long as the tag exists, that commit will exist.

### Triggering the Workflow

Run `make tag-release` to tag the current commit and kick off the workflow.

The release may also be dispatched [manually].

### Workflow Jobs and Process

The workflow consists of a number of concurrently-run jobs, and two final publish jobs.

The publish jobs require manual approval and are only run if the other jobs succeed.

#### `check-version` Job

This job checks that the git ref matches the app version. It matches the ref against the `__version__` variable in `invokeai/version/invokeai_version.py`.

When the workflow is triggered by tag push, the ref is the tag. If the workflow is run manually, the ref is the target selected from the **Use workflow from** dropdown.

This job uses [samuelcolvin/check-python-version].

> Any valid [version specifier] works, so long as the tag matches the version. The release workflow works exactly the same for `RC`, `post`, `dev`, etc.

#### Check and Test Jobs

- **`python-tests`**: runs `pytest` on matrix of platforms
- **`python-checks`**: runs `ruff` (format and lint)
- **`frontend-tests`**: runs `vitest`
- **`frontend-checks`**: runs `prettier` (format), `eslint` (lint), `dpdm` (circular refs), `tsc` (static type check) and `knip` (unused imports)

> **TODO** We should add `mypy` or `pyright` to the **`check-python`** job.

> **TODO** We should add an end-to-end test job that generates an image.

#### `build-installer` Job

This sets up both python and frontend dependencies and builds the python package. Internally, this runs `installer/create_installer.sh` and uploads two artifacts:

- **`dist`**: the python distribution, to be published on PyPI
- **`InvokeAI-installer-${VERSION}.zip`**: the installer to be included in the GitHub release

#### Sanity Check & Smoke Test

At this point, the release workflow pauses as the remaining publish jobs require approval. Time to test the installer.

Because the installer pulls from PyPI, and we haven't published to PyPI yet, you will need to install from the wheel:

- Download and unzip `dist.zip` and the installer from the **Summary** tab of the workflow
- Run the installer script using the `--wheel` CLI arg, pointing at the wheel:

  ```sh
  ./install.sh --wheel ../InvokeAI-4.0.0rc6-py3-none-any.whl
  ```

- Install to a temporary directory so you get the new user experience
- Download a model and generate

> The same wheel file is bundled in the installer and in the `dist` artifact, which is uploaded to PyPI. You should end up with the exactly the same installation as if the installer got the wheel from PyPI.

##### Something isn't right

If testing reveals any issues, no worries. Cancel the workflow, which will cancel the pending publish jobs (you didn't approve them prematurely, right?).

Now you can start from the top:

- Fix the issues and PR the fixes per usual
- Get the PR approved and merged per usual
- Switch to `main` and pull in the fixes
- Run `make tag-release` to move the tag to `HEAD` (which has the fixes) and kick off the release workflow again
- Re-do the sanity check

#### PyPI Publish Jobs

The publish jobs will run if any of the previous jobs fail.

They use [GitHub environments], which are configured as [trusted publishers] on PyPI.

Both jobs require a maintainer to approve them from the workflow's **Summary** tab.

- Click the **Review deployments** button
- Select the environment (either `testpypi` or `pypi`)
- Click **Approve and deploy**

> **If the version already exists on PyPI, the publish jobs will fail.** PyPI only allows a given version to be published once - you cannot change it. If version published on PyPI has a problem, you'll need to "fail forward" by bumping the app version and publishing a followup release.

##### Failing PyPI Publish

Check the [python infrastructure status page] for incidents.

If there are no incidents, contact @hipsterusername or @lstein, who have owner access to GH and PyPI, to see if access has expired or something like that.

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

## Publish the GitHub Release with installer

Once the release is published to PyPI, it's time to publish the GitHub release.

1. [Draft a new release] on GitHub, choosing the tag that triggered the release.
1. Write the release notes, describing important changes. The **Generate release notes** button automatically inserts the changelog and new contributors, and you can copy/paste the intro from previous releases.
1. Use `scripts/get_external_contributions.py` to get a list of external contributions to shout out in the release notes.
1. Upload the zip file created in **`build`** job into the Assets section of the release notes.
1. Check **Set as a pre-release** if it's a pre-release.
1. Check **Create a discussion for this release**.
1. Publish the release.
1. Announce the release in Discord.

> **TODO** Workflows can create a GitHub release from a template and upload release assets. One popular action to handle this is [ncipollo/release-action]. A future enhancement to the release process could set this up.

## Manual Build

The `build installer` workflow can be dispatched manually. This is useful to test the installer for a given branch or tag.

No checks are run, it just builds.

## Manual Release

The `release` workflow can be dispatched manually. You must dispatch the workflow from the right tag, else it will fail the version check.

This functionality is available as a fallback in case something goes wonky. Typically, releases should be triggered via tag push as described above.

[InvokeAI Releases Page]: https://github.com/invoke-ai/InvokeAI/releases
[PyPI]: https://pypi.org/
[Draft a new release]: https://github.com/invoke-ai/InvokeAI/releases/new
[Test PyPI]: https://test.pypi.org/
[version specifier]: https://packaging.python.org/en/latest/specifications/version-specifiers/
[ncipollo/release-action]: https://github.com/ncipollo/release-action
[GitHub environments]: https://docs.github.com/en/actions/deployment/targeting-different-environments/using-environments-for-deployment
[trusted publishers]: https://docs.pypi.org/trusted-publishers/
[samuelcolvin/check-python-version]: https://github.com/samuelcolvin/check-python-version
[manually]: #manual-release
[python infrastructure status page]: https://status.python.org/
