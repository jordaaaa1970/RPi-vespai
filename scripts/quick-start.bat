@echo off
REM VespAI Quick Start Script for Windows
setlocal
set SCRIPT_DIR=%~dp0
for %%I in ("%SCRIPT_DIR%..") do set PROJECT_DIR=%%~fI
cd /d "%PROJECT_DIR%"

echo ========================================
echo VespAI Hornet Detection System
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3 first.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/3] Running automated setup...
python scripts\setup.py
if errorlevel 1 (
    echo ERROR: Setup failed! Check error messages above.
    pause
    exit /b 1
)

echo.
echo [2/3] Setup completed successfully!
echo [3/3] Starting VespAI web interface...
echo.
echo Open your browser to: http://localhost:8081
echo Press Ctrl+C to stop the server
echo.

if exist "%PROJECT_DIR%\start_vespai_web.bat" (
    call "%PROJECT_DIR%\start_vespai_web.bat"
    goto :done
)

if exist "%PROJECT_DIR%\.venv\Scripts\python.exe" (
    "%PROJECT_DIR%\.venv\Scripts\python.exe" vespai.py --web
    goto :done
)

if exist "%PROJECT_DIR%\venv\Scripts\python.exe" (
    "%PROJECT_DIR%\venv\Scripts\python.exe" vespai.py --web
    goto :done
)

python vespai.py --web

:done
pause