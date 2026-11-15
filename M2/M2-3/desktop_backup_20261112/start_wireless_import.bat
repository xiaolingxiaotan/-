@echo off
rem M2-3 wireless inbox auto-import launcher - double click this file on the PC
rem Phone side can use Syncthing / FolderSync / PhotoSync / the web upload page.
python "%~dp0wireless_import.py" %*
if errorlevel 1 pause
