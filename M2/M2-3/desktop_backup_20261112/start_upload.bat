@echo off
rem M2-3 photo upload launcher - double click this file on the PC
python "%~dp0photo_upload.py" %*
if errorlevel 1 pause
