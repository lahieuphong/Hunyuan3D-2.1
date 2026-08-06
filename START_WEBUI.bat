@echo off
chcp 65001 >nul
setlocal

rem 始终从 Hunyuan3D 仓库目录运行，即使通过双击启动。
cd /d "%~dp0"

set "WEBUI_SCRIPT=%~dp0hy3dshape\scripts\start_windows_multiview_webui.ps1"
set "PYTHON_EXE=%~dp0.venv-win\Scripts\python.exe"

title 混元3D 四视图网页界面

if not exist "%WEBUI_SCRIPT%" (
    echo [错误] 找不到网页界面启动脚本：
    echo %WEBUI_SCRIPT%
    echo.
    echo 按任意键关闭此窗口。
    pause >nul
    exit /b 1
)

if not exist "%PYTHON_EXE%" (
    echo [错误] 找不到 Windows Python 环境：
    echo %PYTHON_EXE%
    echo.
    echo 应存在的目录：.venv-win
    echo.
    echo 按任意键关闭此窗口。
    pause >nul
    exit /b 1
)

echo 正在后台启动混元3D 网页界面...
echo 模型准备就绪后将自动打开浏览器。
echo 运行日志：hy3dshape\output_folder\webui\logs
echo.

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%WEBUI_SCRIPT%" -Background -OpenBrowser
set "WEBUI_EXIT_CODE=%ERRORLEVEL%"

if not "%WEBUI_EXIT_CODE%"=="0" (
    echo.
    echo [错误] 混元3D 网页界面启动失败。
    echo 退出代码：%WEBUI_EXIT_CODE%
    echo 请保持此窗口打开，并发送错误信息以便排查。
    echo.
    echo 按任意键关闭此窗口。
    pause >nul
    exit /b %WEBUI_EXIT_CODE%
)

exit /b 0
