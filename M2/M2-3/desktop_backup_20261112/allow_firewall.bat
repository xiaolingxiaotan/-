@echo off
rem Right-click -> "Run as administrator". Opens inbound TCP 8000 for wireless_upload.py.
net session >nul 2>&1
if errorlevel 1 (
  echo Please right-click this file and choose "Run as administrator".
  pause
  exit /b 1
)
netsh advfirewall firewall delete rule name="M23 Photo Upload TCP 8000" >nul 2>&1
netsh advfirewall firewall add rule name="M23 Photo Upload TCP 8000" dir=in action=allow protocol=TCP localport=8000 profile=any
echo.
echo Done. Now double-click start_all.bat and use the URL printed in its first window.
pause
