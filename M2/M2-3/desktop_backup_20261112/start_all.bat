@echo off
rem M2-3 one-click: opens the wireless inbox server AND the auto-importer.
rem Two console windows will appear. Keep both open while shooting.
start "M23 Inbox Server" cmd /k python "%~dp0wireless_upload.py"
start "M23 Auto Import" cmd /k python "%~dp0wireless_import.py"
