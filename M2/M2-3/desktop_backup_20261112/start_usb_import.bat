@echo off
rem M2-3 USB auto-import launcher - double click this file on the PC
rem No network needed. Connect the phone by USB cable and choose "File transfer".
python "%~dp0usb_import.py" %*
if errorlevel 1 pause
