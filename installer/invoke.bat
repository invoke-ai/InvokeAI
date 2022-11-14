@echo off

call .venv\Scripts\activate.bat

echo Do you want to generate images using the
echo 1. command-line
echo 2. browser-based UI
echo 3. open the developer console
set /P restore="Please enter 1, 2 or 3: "
IF /I "%restore%" == "1" (
    echo Starting the InvokeAI command-line..
    .venv\Scripts\python scripts\invoke.py
) ELSE IF /I "%restore%" == "2" (
    echo Starting the InvokeAI browser-based UI..
    .venv\Scripts\python scripts\invoke.py --web
) ELSE IF /I "%restore%" == "3" (
    echo Developer Console
    echo 'python' command is:
    where python
    echo 'python' version is:
    python --version
    echo Type 'exit' to quit this shell
    call cmd
) ELSE (
    echo Invalid selection
    pause
    exit /b
)

deactivate
