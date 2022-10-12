@echo off

@rem prevent the window from closing after running the commands
if not defined in_subprocess (cmd /k set in_subprocess=y ^& %0 %*) & exit )

set INSTALL_ENV_DIR=%cd%\installer_files\env
set PATH=%PATH%;%INSTALL_ENV_DIR%;%INSTALL_ENV_DIR%\Library\bin;%INSTALL_ENV_DIR%\Scripts

call conda activate invokeai

echo Ready to dream..
