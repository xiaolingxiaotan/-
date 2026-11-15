@echo off
rem M2-3 wireless inbox server - WebDAV + phone browser rapid capture
rem Double click on the PC, then open the printed URL on the phone.
python "%~dp0wireless_upload.py" %*
if errorlevel 1 pause
