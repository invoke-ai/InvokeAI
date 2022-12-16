---
title: build binary installers
---

# :simple-buildkite: How to build "binary" installers (InvokeAI-mac/windows/linux_on_*.zip)

## 1. Ensure `installers/requirements.in` is correct

and up to date on the branch to be installed.

## <a name="step-2"></a> 2. Run `pip-compile` on each platform.

On each target platform, in the branch that is to be installed, and
inside the InvokeAI git root folder, run the following commands:

```commandline
conda activate invokeai # or however you activate python
pip install pip-tools
pip-compile --allow-unsafe --generate-hashes --output-file=binary_installer/<reqsfile>.txt binary_installer/requirements.in
```
where `<reqsfile>.txt` is whichever of
```commandline
py3.10-darwin-arm64-mps-reqs.txt
py3.10-darwin-x86_64-reqs.txt
py3.10-linux-x86_64-cuda-reqs.txt
py3.10-windows-x86_64-cuda-reqs.txt
```
matches the current OS and architecture.
> There is no way to cross-compile these. They must be done on a system matching the target OS and arch.

## <a name="step-3"></a> 3. Set github repository and branch

Once all reqs files have been collected and committed **to the branch
to be installed**, edit `binary_installer/install.sh.in` and `binary_installer/install.bat.in` so that `RELEASE_URL`
and `RELEASE_SOURCEBALL` point to the github repo and branch that is
to be installed.

For example, to install `main` branch of `InvokeAI`, they should be
set as follows:

`install.sh.in`:
```commandline
RELEASE_URL=https://github.com/invoke-ai/InvokeAI
RELEASE_SOURCEBALL=/archive/refs/heads/main.tar.gz
```

`install.bat.in`:
```commandline
set RELEASE_URL=https://github.com/invoke-ai/InvokeAI
set RELEASE_SOURCEBALL=/archive/refs/heads/main.tar.gz
```

Or, to install `damians-cool-feature` branch of `damian0815`, set them
as follows:

`install.sh.in`:
```commandline
RELEASE_URL=https://github.com/damian0815/InvokeAI
RELEASE_SOURCEBALL=/archive/refs/heads/damians-cool-feature.tar.gz
```

`install.bat.in`:
```commandline
set RELEASE_URL=https://github.com/damian0815/InvokeAI
set RELEASE_SOURCEBALL=/archive/refs/heads/damians-cool-feature.tar.gz
```

The branch and repo specified here **must** contain the correct reqs
files. The installer zip files **do not** contain requirements files,
they are pulled from the specified branch during the installation
process.

## 4. Create zip files.

cd into the `installers/` folder and run
`./create_installers.sh`. This will create
`InvokeAI-mac_on_<branch>.zip`,
`InvokeAI-windows_on_<branch>.zip` and
`InvokeAI-linux_on_<branch>.zip`. These files can be distributed to end users.

These zips will continue to function as installers for all future
pushes to those branches, as long as necessary changes to
`requirements.in` are propagated in a timely manner to the
`py3.10-*-reqs.txt` files using pip-compile as outlined in [step
2](#step-2).

To actually install, users should unzip the appropriate zip file into an empty
folder and run `install.sh` on macOS/Linux or `install.bat` on
Windows.
