@echo off
title MayaJaal - Platform Launcher
echo ======================================================================
echo           MAYAJAAL / SIH26187 - LAUNCH ALL SERVICES
echo ======================================================================
echo.
echo [*] Starting Backend Server (FastAPI on :8000)...
start "MayaJaal Backend" "%~dp0start_backend.bat"

echo [*] Starting Frontend Server (Next.js on :3000)...
start "MayaJaal Frontend" "%~dp0start_frontend.bat"

echo.
echo ======================================================================
echo [*] Both services launched in separate windows!
echo [*] Backend API:       http://localhost:8000
echo [*] Swagger API Docs:  http://localhost:8000/docs
echo [*] Frontend Web UI:   http://localhost:3000
echo ======================================================================
echo.
timeout /t 5 >nul
