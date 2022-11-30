@echo off

call .venv\Scripts\activate.bat

echo Do you want to generate images using the
echo 1. command-line
echo 2. browser-based UI
echo OR
echo 3. open the developer console
set /p choice="Please enter 1, 2 or 3: "
if /i "%choice%" == "1" (
    echo Starting the InvokeAI command-line.
    .venv\Scripts\python scripts\invoke.py
) else if /i "%choice%" == "2" (
    echo Starting the InvokeAI browser-based UI.
    .venv\Scripts\python scripts\invoke.py --web
) else if /i "%choice%" == "3" (
    echo Developer Console
    echo Python command is:
    where python
    echo Python version is:
    python --version
    echo *************************
    echo You are now in the system shell, with the local InvokeAI Python virtual environment activated,
    echo so that you can troubleshoot this InvokeAI installation as necessary.
    echo *************************
    echo *** Type `exit` to quit this shell and deactivate the Python virtual environment ***
    call cmd /k
) else (
    echo Invalid selection
    pause
    exit /b
)

deactivate
