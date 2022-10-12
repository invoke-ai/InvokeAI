@echo off

@rem prevent the window from closing after running the commands
if not defined in_subprocess (cmd /k set in_subprocess=y ^& %0 %*) & exit )

@rem check if conda exists, otherwise use micromamba
set CONDA_COMMAND=conda

call conda --version "" >tmp/stdout.txt 2>tmp/stderr.txt
if "%ERRORLEVEL%" NEQ "0" set CONDA_COMMAND=micromamba

@rem initialize micromamba, if using that
if "%CONDA_COMMAND%" EQU "micromamba" (
    set MAMBA_ROOT_PREFIX=%cd%\installer_files\mamba
    set INSTALL_ENV_DIR=%cd%\installer_files\env

    if not exist "%MAMBA_ROOT_PREFIX%\condabin" (
        echo "Have you run install.bat?"
        exit /b
    )

    call "%MAMBA_ROOT_PREFIX%\condabin\mamba_hook.bat"

    call micromamba activate "%INSTALL_ENV_DIR%"
)

call %CONDA_COMMAND% activate invokeai

pause
