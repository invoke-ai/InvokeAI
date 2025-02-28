# Release Process

The Invoke application is published as a python package on [PyPI]. This includes both a source distribution and built distribution (a wheel).

Most users install it with the [Launcher](https://github.com/invoke-ai/launcher/), others with `pip`.

The launcher uses GitHub as the source of truth for available releases.

## Broad Strokes

- Merge all changes and bump the version in the codebase.
- Tag the release commit.
- Wait for the release workflow to complete.
- Approve the PyPI publish jobs.
- Write GH release notes.

## General Prep

Make a developer call-out for PRs to merge. Merge and test things out. Bump the version by editing `invokeai/version/invokeai_version.py`.

## Release Workflow

The `release.yml` workflow runs a number of jobs to handle code checks, tests, build and publish on PyPI.

It is triggered on **tag push**, when the tag matches `v*`.

### Triggering the Workflow

Ensure all commits that should be in the release are merged, and you have pulled them locally.

Double-check that you have checked out the commit that will represent the release (typically the latest commit on `main`).

Run `make tag-release` to tag the current commit and kick off the workflow. You will be prompted to provide a message - use the version specifier.

If this version's tag already exists for some reason (maybe you had to make a last minute change), the script will overwrite it.

> In case you cannot use the Make target, the release may also be dispatched [manually] via GH.

### Workflow Jobs and Process

The workflow consists of a number of concurrently-run checks and tests, then two final publish jobs.

The publish jobs require manual approval and are only run if the other jobs succeed.

#### `check-version` Job

This job ensures that the `invokeai` python package version specifier matches the tag for the release. The version specifier is pulled from the `__version__` variable in `invokeai/version/invokeai_version.py`.

This job uses [samuelcolvin/check-python-version].

> Any valid [version specifier] works, so long as the tag matches the version. The release workflow works exactly the same for `RC`, `post`, `dev`, etc.

#### Check and Test Jobs

Next, these jobs run and must pass. They are the same jobs that are run for every PR.

- **`python-tests`**: runs `pytest` on matrix of platforms
- **`python-checks`**: runs `ruff` (format and lint)
- **`frontend-tests`**: runs `vitest`
- **`frontend-checks`**: runs `prettier` (format), `eslint` (lint), `dpdm` (circular refs), `tsc` (static type check) and `knip` (unused imports)
- **`typegen-checks`**: ensures the frontend and backend types are synced

#### `build-installer` Job

This sets up both python and frontend dependencies and builds the python package. Internally, this runs `installer/create_installer.sh` and uploads two artifacts:

- **`dist`**: the python distribution, to be published on PyPI
- **`InvokeAI-installer-${VERSION}.zip`**: the legacy install scripts

You don't need to download either of these files.

> The legacy install scripts are no longer used, but we haven't updated the workflow to skip building them.

#### Sanity Check & Smoke Test

At this point, the release workflow pauses as the remaining publish jobs require approval.

It's possible to test the python package before it gets published to PyPI. We've never had problems with it, so it's not necessary to do this.

But, if you want to be extra-super careful, here's how to test it:

- Download the `dist.zip` build artifact from the `build-installer` job
- Unzip it and find the wheel file
- Create a fresh Invoke install by following the [manual install guide](https://invoke-ai.github.io/InvokeAI/installation/manual/) - but instead of installing from PyPI, install from the wheel
- Test the app

##### Something isn't right

If testing reveals any issues, no worries. Cancel the workflow, which will cancel the pending publish jobs (you didn't approve them prematurely, right?) and start over.

#### PyPI Publish Jobs

The publish jobs will not run if any of the previous jobs fail.

They use [GitHub environments], which are configured as [trusted publishers] on PyPI.

Both jobs require a @hipsterusername or @psychedelicious to approve them from the workflow's **Summary** tab.

- Click the **Review deployments** button
- Select the environment (either `testpypi` or `pypi` - typically you select both)
- Click **Approve and deploy**

> **If the version already exists on PyPI, the publish jobs will fail.** PyPI only allows a given version to be published once - you cannot change it. If version published on PyPI has a problem, you'll need to "fail forward" by bumping the app version and publishing a followup release.

##### Failing PyPI Publish

Check the [python infrastructure status page] for incidents.

If there are no incidents, contact @hipsterusername or @lstein, who have owner access to GH and PyPI, to see if access has expired or something like that.

#### `publish-testpypi` Job

Publishes the distribution on the [Test PyPI] index, using the `testpypi` GitHub environment.

This job is not required for the production PyPI publish, but included just in case you want to test the PyPI release for some reason:

- Approve this publish job without approving the prod publish
- Let it finish
- Create a fresh Invoke install by following the [manual install guide](https://invoke-ai.github.io/InvokeAI/installation/manual/), making sure to use the Test PyPI index URL: `https://test.pypi.org/simple/`
- Test the app

#### `publish-pypi` Job

Publishes the distribution on the production PyPI index, using the `pypi` GitHub environment.

It's a good idea to wait to approve and run this job until you have the release notes ready!

## Prep and publish the GitHub Release

1. [Draft a new release] on GitHub, choosing the tag that triggered the release.
2. The **Generate release notes** button automatically inserts the changelog and new contributors. Make sure to select the correct tags for this release and the last stable release. GH often selects the wrong tags - do this manually.
3. Write the release notes, describing important changes. Contributions from community members should be shouted out. Use the GH-generated changelog to see all contributors. If there are Weblate translation updates, open that PR and shout out every person who contributed a translation.
4. Check **Set as a pre-release** if it's a pre-release.
5. Approve and wait for the `publish-pypi` job to finish if you haven't already.
6. Publish the GH release.
7. Post the release in Discord in the [releases](https://discord.com/channels/1020123559063990373/1149260708098359327) channel with abbreviated notes. For example:
   > Invoke v5.7.0 (stable): <https://github.com/invoke-ai/InvokeAI/releases/tag/v5.7.0>
   >
   > It's a pretty big one - Form Builder, Metadata Nodes (thanks @SkunkWorxDark!), and much more.
8. Right click the message in releases and copy the link to it. Then, post that link in the [new-release-discussion](https://discord.com/channels/1020123559063990373/1149506274971631688) channel. For example:
   > Invoke v5.7.0 (stable): <https://discord.com/channels/1020123559063990373/1149260708098359327/1344521744916021248>

## Manual Release

The `release` workflow can be dispatched manually. You must dispatch the workflow from the right tag, else it will fail the version check.

This functionality is available as a fallback in case something goes wonky. Typically, releases should be triggered via tag push as described above.

[PyPI]: https://pypi.org/
[Draft a new release]: https://github.com/invoke-ai/InvokeAI/releases/new
[Test PyPI]: https://test.pypi.org/
[version specifier]: https://packaging.python.org/en/latest/specifications/version-specifiers/
[GitHub environments]: https://docs.github.com/en/actions/deployment/targeting-different-environments/using-environments-for-deployment
[trusted publishers]: https://docs.pypi.org/trusted-publishers/
[samuelcolvin/check-python-version]: https://github.com/samuelcolvin/check-python-version
[manually]: #manual-release
[python infrastructure status page]: https://status.python.org/
