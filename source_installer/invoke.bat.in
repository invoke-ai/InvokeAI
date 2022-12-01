@echo off

set INSTALL_ENV_DIR=%cd%\installer_files\env
set PATH=%INSTALL_ENV_DIR%;%INSTALL_ENV_DIR%\Library\bin;%INSTALL_ENV_DIR%\Scripts;%INSTALL_ENV_DIR%\Library\usr\bin;%PATH%

call conda activate invokeai

echo Do you want to generate images using the
echo 1. command-line
echo 2. browser-based UI
echo 3. open the developer console
set /P restore="Please enter 1, 2 or 3: "
IF /I "%restore%" == "1" (
    echo Starting the InvokeAI command-line..
    python scripts\invoke.py
) ELSE IF /I "%restore%" == "2" (
    echo Starting the InvokeAI browser-based UI..
    python scripts\invoke.py --web
) ELSE IF /I "%restore%" == "3" (
    echo Developer Console
    call where python
    call python --version

    cmd /k
) ELSE (
    echo Invalid selection
    pause
    exit /b
)
