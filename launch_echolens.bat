@echo off
title Echolens — Smart-Glasses AI Acoustic Hub
cd /d "%~dp0"

echo ================================================================
echo  👓 Echolens — Smart-Glasses Target Speech Extraction (CHiME-10)
echo ================================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found in .venv!
    echo Please ensure the .venv folder exists in %CD%.
    pause
    exit /b 1
)

echo [1/3] Activating virtual environment (.venv)...
call .\.venv\Scripts\activate.bat

echo [2/3] Opening Echolens Web Dashboard in your default browser...
start "" http://localhost:8000

echo [3/3] Starting Echolens server on 0.0.0.0:8000...
echo.
python -m echolens server --host 0.0.0.0 --port 8000

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Server exited with code %ERRORLEVEL%.
    pause
)
