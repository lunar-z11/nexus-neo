@echo off
setlocal
cd /d "%~dp0"
python src/main.py %*
