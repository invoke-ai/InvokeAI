@echo off

@rem This script will install git and conda (if not found on the PATH variable)
@rem  using micromamba (an 8mb static-linked single-file binary, conda replacement).
@rem For users who already have git and conda, this step will be skipped.

@rem Next, it'll checkout the project's git repo, if necessary.
@rem Finally, it'll create the conda environment and preload the models.

@rem This enables a user to install this project without manually installing conda and git.

@rem prevent the window from closing after an error
if not defined in_subprocess (cmd /k set in_subprocess=y ^& %0 %*) & exit )

@rem config
set MAMBA_ROOT_PREFIX=%cd%\installer_files\mamba
set INSTALL_ENV_DIR=%cd%\installer_files\env
set MICROMAMBA_BINARY_FILE=%cd%\installer_files\micromamba_win_x64.exe

@rem figure out whether git and conda needs to be installed
set PACKAGES_TO_INSTALL=

call conda --version "" >tmp/stdout.txt 2>tmp/stderr.txt
if "%ERRORLEVEL%" NEQ "0" set PACKAGES_TO_INSTALL=%PACKAGES_TO_INSTALL% conda

call git --version "" >tmp/stdout.txt 2>tmp/stderr.txt
if "%ERRORLEVEL%" NEQ "0" set PACKAGES_TO_INSTALL=%PACKAGES_TO_INSTALL% git

@rem (if necessary) install git and conda into a contained environment
if "%PACKAGES_TO_INSTALL%" NEQ "" (
    @rem initialize micromamba
    if not exist "%MAMBA_ROOT_PREFIX%" (
        mkdir "%MAMBA_ROOT_PREFIX%"
        copy "%MICROMAMBA_BINARY_FILE%" "%MAMBA_ROOT_PREFIX%\micromamba.exe"

        @rem test the mamba binary
        echo Micromamba version:
        call "%MAMBA_ROOT_PREFIX%\micromamba.exe" --version
    )

    @rem create the installer env
    if not exist "%INSTALL_ENV_DIR%" (
        call "%MAMBA_ROOT_PREFIX%\micromamba.exe" create -y --prefix "%INSTALL_ENV_DIR%"
    )

    echo "Packages to install:%PACKAGES_TO_INSTALL%"

    call "%MAMBA_ROOT_PREFIX%\micromamba.exe" install -y --prefix "%INSTALL_ENV_DIR%" -c conda-forge %PACKAGES_TO_INSTALL%
)

set PATH=%PATH%;%INSTALL_ENV_DIR%;%INSTALL_ENV_DIR%\Library\bin;%INSTALL_ENV_DIR%\Scripts

@rem get the repo (and load into the current directory)
if not exist ".git" (
    call git config --global init.defaultBranch main
    call git init
    call git remote add origin https://github.com/cmdr2/InvokeAI.git
    call git fetch
    call git checkout origin/main -ft
)

@rem create the environment
call conda env create
call conda activate invokeai

@rem preload the models
call python scripts\preload_models.py

@rem make the models dir
mkdir models\ldm\stable-diffusion-v1

@rem tell the user that they need to download the ckpt
echo.
echo "Now you need to install the weights for the stable diffusion model."
echo "Please follow the steps at https://invoke-ai.github.io/InvokeAI/installation/INSTALL_WINDOWS/ to complete the installation"

@rem it would be nice if the weights downloaded automatically, and didn't need the user to do this manually.

pause
