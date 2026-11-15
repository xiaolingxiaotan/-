@echo off
title M2-3 Label Web
echo.
echo   Starting M2-3 annotation server (browser version)...
echo   The browser will open automatically in a few seconds.
echo   Keep this window open while labeling. Close it / press Ctrl+C to stop.
echo.
wsl.exe -d Ubuntu2204 -e bash /home/xiaolingxiaotan/run_label_web.sh
if not errorlevel 1 exit /b 0
echo.
echo   Startup failed. Run this in a WSL terminal to see the reason:
echo       python3 label_web.py
echo.
pause
