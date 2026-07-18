@echo off
setlocal

rem Always run from the Hunyuan3D repository, even when opened by double-click.
cd /d "%~dp0"

set "WEBUI_SCRIPT=%~dp0hy3dshape\scripts\start_windows_multiview_webui.ps1"
set "PYTHON_EXE=%~dp0.venv-win\Scripts\python.exe"

title Hunyuan3D 4-View Web UI

if not exist "%WEBUI_SCRIPT%" (
    echo [ERROR] Cannot find the Web UI launcher:
    echo %WEBUI_SCRIPT%
    echo.
    pause
    exit /b 1
)

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Cannot find the Windows Python environment:
    echo %PYTHON_EXE%
    echo.
    echo Expected folder: .venv-win
    pause
    exit /b 1
)

echo Starting Hunyuan3D Web UI in the background...
echo The browser will open automatically after the model is ready.
echo Runtime logs: hy3dshape\output_folder\webui\logs
echo.

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%WEBUI_SCRIPT%" -Background -OpenBrowser
set "WEBUI_EXIT_CODE=%ERRORLEVEL%"

if not "%WEBUI_EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] Hunyuan3D Web UI could not start.
    echo Exit code: %WEBUI_EXIT_CODE%
    echo Keep this window open and send the error text for inspection.
    echo.
    pause
    exit /b %WEBUI_EXIT_CODE%
)

exit /b 0
