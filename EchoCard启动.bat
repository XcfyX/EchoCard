@echo off
rem EchoCard 一键启动（桌面窗口 + 托盘常驻）
rem 若已有一个 EchoCard 在运行，请先通过托盘菜单「完全退出」
cd /d "D:\Projects\EchoCard"
start "EchoCard" ".venv\Scripts\pythonw.exe" -m echocard
exit
