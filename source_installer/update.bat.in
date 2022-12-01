@echo off

set INSTALL_ENV_DIR=%cd%\installer_files\env
set PATH=%INSTALL_ENV_DIR%;%INSTALL_ENV_DIR%\Library\bin;%INSTALL_ENV_DIR%\Scripts;%INSTALL_ENV_DIR%\Library\usr\bin;%PATH%

@rem update the repo
if exist ".git" (
    call git pull
)


conda env update
conda activate invokeai
python scripts/preload_models.py

echo "Press any key to continue"
pause
exit 0

