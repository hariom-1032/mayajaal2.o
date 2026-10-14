@echo off
setlocal enabledelayedexpansion
title MayaJaal - Backend Inference Engine

echo ======================================================================
echo           MAYAJAAL / SIH26187 - INFERENCE ENGINE BACKEND
echo ======================================================================
echo.

:: Determine root and backend paths
set "SCRIPT_DIR=%~dp0"
if exist "%SCRIPT_DIR%app\main.py" (
    set "BACKEND_DIR=%SCRIPT_DIR%"
) else if exist "%SCRIPT_DIR%backend\app\main.py" (
    set "BACKEND_DIR=%SCRIPT_DIR%backend"
) else (
    echo [ERROR] Could not locate backend directory with app\main.py.
    echo Please make sure this script is in the project root or backend folder.
    echo.
    pause
    exit /b 1
)

cd /d "%BACKEND_DIR%"
echo [*] Working directory: %BACKEND_DIR%

:: Detect Python executable / Virtual Environment
set "PYTHON_EXE="

if exist "%BACKEND_DIR%\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%BACKEND_DIR%\.venv\Scripts\python.exe"
    echo [*] Using backend virtual environment: %BACKEND_DIR%\.venv
) else if exist "%BACKEND_DIR%\venv\Scripts\python.exe" (
    set "PYTHON_EXE=%BACKEND_DIR%\venv\Scripts\python.exe"
    echo [*] Using backend virtual environment: %BACKEND_DIR%\venv
) else if exist "%SCRIPT_DIR%\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%SCRIPT_DIR%\.venv\Scripts\python.exe"
    echo [*] Using root virtual environment: %SCRIPT_DIR%\.venv
) else (
    where python >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        set "PYTHON_EXE=python"
        echo [*] Using system Python from PATH
    ) else (
        echo [ERROR] Python was not found in PATH or in a virtual environment.
        echo Please install Python 3.10+ and ensure it is added to PATH.
        echo.
        pause
        exit /b 1
    )
)

:: Verify python runs
"%PYTHON_EXE%" --version >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to run "%PYTHON_EXE%". Please check your Python installation.
    pause
    exit /b 1
)

:: Check if uvicorn is installed
"%PYTHON_EXE%" -c "import uvicorn" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [WARNING] 'uvicorn' is not installed in the selected Python environment.
    echo [*] Attempting to install required packages from requirements.txt...
    "%PYTHON_EXE%" -m pip install -r requirements.txt
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to install dependencies. Please run 'pip install -r requirements.txt' manually.
        pause
        exit /b 1
    )
)

:: Check if .env file exists
if not exist "%BACKEND_DIR%\.env" (
    if exist "%BACKEND_DIR%\.env.example" (
        echo [*] .env not found. Creating .env from .env.example...
        copy "%BACKEND_DIR%\.env.example" "%BACKEND_DIR%\.env" >nul
    )
)

echo.
echo [*] Starting FastAPI Backend on http://localhost:8000 ...
echo [*] Interactive API docs will be at: http://localhost:8000/docs
echo [*] Press Ctrl+C to stop the server.
echo ======================================================================
echo.

"%PYTHON_EXE%" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo [!] Server stopped with error code %ERRORLEVEL%.
    pause
)
