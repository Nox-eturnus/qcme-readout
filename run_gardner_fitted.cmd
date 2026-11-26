@echo off
rem Fitted-box readout factors of the Gardner toggle (about five hours). Run this file from your own terminal.
rem The result is saved to results\readout\fitted_gardner.json, the log to outputs\logs\fitted_gardner.log.
cd /d "%~dp0"
if not exist outputs\logs mkdir outputs\logs
start "gardner fitted" /min cmd /c ""C:\Users\Aarush Das Bansiwal\AppData\Local\Programs\Python\Python313\python.exe" -m analysis.readout fitted_gardner > outputs\logs\fitted_gardner.log 2> outputs\logs\fitted_gardner.err"
