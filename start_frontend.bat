@echo off
title MayaJaal - Frontend Web UI
echo ======================================================================
echo           MAYAJAAL / SIH26187 - FRONTEND WEB UI
echo ======================================================================
echo.
cd /d "%~dp0frontend"
echo [*] Working directory: %cd%
echo [*] Starting Next.js Dev Server on http://localhost:3000 ...
echo [*] Press Ctrl+C to stop.
echo ======================================================================
echo.
call npm.cmd run dev
if %ERRORLEVEL% neq 0 (
    echo.
    echo [!] Frontend server stopped with error code %ERRORLEVEL%.
    pause
)
