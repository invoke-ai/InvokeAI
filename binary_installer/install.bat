@echo off

@rem This script will install git (if not found on the PATH variable)
@rem  using micromamba (an 8mb static-linked single-file binary, conda replacement).
@rem For users who already have git, this step will be skipped.

@rem Next, it'll download the project's source code.
@rem Then it will download a self-contained, standalone Python and unpack it.
@rem Finally, it'll create the Python virtual environment and preload the models.

@rem This enables a user to install this project without manually installing git or Python

set "no_cache_dir=--no-cache-dir"
if "%1" == "use-cache" (
    set "no_cache_dir="
)

echo ***** Installing InvokeAI.. *****
echo "USING development BRANCH. REMEMBER TO CHANGE TO main BEFORE RELEASE"
@rem Config
set INSTALL_ENV_DIR=%cd%\installer_files\env
@rem https://mamba.readthedocs.io/en/latest/installation.html
set MICROMAMBA_DOWNLOAD_URL=https://github.com/cmdr2/stable-diffusion-ui/releases/download/v1.1/micromamba.exe
set RELEASE_URL=https://github.com/invoke-ai/InvokeAI
#set RELEASE_SOURCEBALL=/archive/refs/heads/main.tar.gz
# RELEASE_SOURCEBALL=/archive/refs/heads/test-installer.tar.gz
RELEASE_SOURCEBALL=/archive/refs/heads/development.tar.gz
set PYTHON_BUILD_STANDALONE_URL=https://github.com/indygreg/python-build-standalone/releases/download
set PYTHON_BUILD_STANDALONE=20221002/cpython-3.10.7+20221002-x86_64-pc-windows-msvc-shared-install_only.tar.gz

set PACKAGES_TO_INSTALL=

call git --version >.tmp1 2>.tmp2
if "%ERRORLEVEL%" NEQ "0" set PACKAGES_TO_INSTALL=%PACKAGES_TO_INSTALL% git

@rem Cleanup
del /q .tmp1 .tmp2

@rem (if necessary) install git into a contained environment
if "%PACKAGES_TO_INSTALL%" NEQ "" (
    @rem download micromamba
    echo ***** Downloading micromamba from %MICROMAMBA_DOWNLOAD_URL% to micromamba.exe *****

    call curl -L "%MICROMAMBA_DOWNLOAD_URL%" > micromamba.exe

    @rem test the mamba binary
    echo ***** Micromamba version:  *****
    call micromamba.exe --version

    @rem create the installer env
    if not exist "%INSTALL_ENV_DIR%" (
        call micromamba.exe create -y --prefix "%INSTALL_ENV_DIR%"
    )

    echo ***** Packages to install:%PACKAGES_TO_INSTALL% *****

    call micromamba.exe install -y --prefix "%INSTALL_ENV_DIR%" -c conda-forge %PACKAGES_TO_INSTALL%

    if not exist "%INSTALL_ENV_DIR%" (
        echo ----- There was a problem while installing "%PACKAGES_TO_INSTALL%" using micromamba. Cannot continue. -----
        pause
        exit /b
    )
)

del /q micromamba.exe

@rem For 'git' only
set PATH=%INSTALL_ENV_DIR%\Library\bin;%PATH%

@rem Download/unpack/clean up InvokeAI release sourceball
set err_msg=----- InvokeAI source download failed -----
echo Trying to download "%RELEASE_URL%%RELEASE_SOURCEBALL%"
curl -L %RELEASE_URL%%RELEASE_SOURCEBALL% --output InvokeAI.tgz
if %errorlevel% neq 0 goto err_exit

set err_msg=----- InvokeAI source unpack failed -----
tar -zxf InvokeAI.tgz
if %errorlevel% neq 0 goto err_exit

del /q InvokeAI.tgz

set err_msg=----- InvokeAI source copy failed -----
cd InvokeAI-*
xcopy . .. /e /h
if %errorlevel% neq 0 goto err_exit
cd ..

@rem cleanup
for /f %%i in ('dir /b InvokeAI-*') do rd /s /q %%i
rd /s /q .dev_scripts .github docker-build tests
del /q requirements.in requirements-mkdocs.txt shell.nix

echo ***** Unpacked InvokeAI source *****

@rem Download/unpack/clean up python-build-standalone
set err_msg=----- Python download failed -----
curl -L %PYTHON_BUILD_STANDALONE_URL%/%PYTHON_BUILD_STANDALONE% --output python.tgz
if %errorlevel% neq 0 goto err_exit

set err_msg=----- Python unpack failed -----
tar -zxf python.tgz
if %errorlevel% neq 0 goto err_exit

del /q python.tgz

echo ***** Unpacked python-build-standalone *****

@rem create venv
set err_msg=----- problem creating venv -----
.\python\python -E -s -m venv .venv
if %errorlevel% neq 0 goto err_exit
call .venv\Scripts\activate.bat

echo ***** Created Python virtual environment *****

@rem Print venv's Python version
set err_msg=----- problem calling venv's python -----
echo We're running under
.venv\Scripts\python --version
if %errorlevel% neq 0 goto err_exit

set err_msg=----- pip update failed -----
.venv\Scripts\python -m pip install %no_cache_dir% --no-warn-script-location --upgrade pip wheel
if %errorlevel% neq 0 goto err_exit

echo ***** Updated pip and wheel *****

set err_msg=----- requirements file copy failed -----
copy installer\py3.10-windows-x86_64-cuda-reqs.txt requirements.txt
if %errorlevel% neq 0 goto err_exit

set err_msg=----- main pip install failed -----
.venv\Scripts\python -m pip install %no_cache_dir% --no-warn-script-location -r requirements.txt
if %errorlevel% neq 0 goto err_exit

echo ***** Installed Python dependencies *****

set err_msg=----- InvokeAI setup failed -----
.venv\Scripts\python -m pip install %no_cache_dir% --no-warn-script-location -e .
if %errorlevel% neq 0 goto err_exit

copy installer\invoke.bat .\invoke.bat
echo ***** Installed invoke launcher script ******

@rem more cleanup
rd /s /q installer installer_files

@rem preload the models
call .venv\Scripts\python scripts\configure_invokeai.py
set err_msg=----- model download clone failed -----
if %errorlevel% neq 0 goto err_exit
deactivate

echo ***** Finished downloading models *****

echo All done! Execute the file invoke.bat in this directory to start InvokeAI
pause
exit

:err_exit
    echo %err_msg%
    pause
    exit
