@echo off

@rem prevent the window from closing after an error
if not defined in_subprocess (cmd /k set in_subprocess=y ^& %0 %*) & exit )

set INSTALL_ENV_DIR=%cd%\installer_files\env
set PATH=%INSTALL_ENV_DIR%;%INSTALL_ENV_DIR%\Library\bin;%INSTALL_ENV_DIR%\Scripts;%PATH%

@rem update the repo
if exist ".git" (
    call git pull
)

conda env update
