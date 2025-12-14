@echo off
REM SEMAI ML Pipeline Runner for Windows
setlocal enabledelayedexpansion

cd /d c:\Users\ACER\.crewai\semai
set PYTHONPATH=c:\Users\ACER\.crewai\semai\src

REM Use Conda base environment which has kagglehub installed
call conda run -p C:\Users\ACER\anaconda3 python src/semai/main.py

pause
