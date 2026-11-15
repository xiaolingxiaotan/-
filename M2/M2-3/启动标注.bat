@echo off
chcp 65001
title M2-3 Label Tool
echo Starting M2-3 label tool (WSL)...
wsl.exe -d Ubuntu2204 -e bash /home/xiaolingxiaotan/run_label_tool.sh
if not errorlevel 1 exit /b 0
echo.
echo Startup failed. Last lines of log:
wsl.exe -d Ubuntu2204 -e bash /home/xiaolingxiaotan/run_label_tool.sh --log
echo.
pause
