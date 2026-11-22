@echo off
rem EchoCard launcher: desktop window + tray. Quit via tray menu if already running.
cd /d "D:\Projects\EchoCard"
start "EchoCard" ".venv\Scripts\pythonw.exe" -m echocard
exit
