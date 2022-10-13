@echo off

set INSTALL_ENV_DIR=%cd%\installer_files\env
set PATH=%INSTALL_ENV_DIR%;%INSTALL_ENV_DIR%\Library\bin;%INSTALL_ENV_DIR%\Scripts;%PATH%

call conda activate invokeai

echo Ready to dream..

cmd /k
