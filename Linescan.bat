@echo off
set base=%localappdata%\Continuum\miniconda3
REM echo %base%
set script="C:\Users\sylvain.finot\Cloud CNRS\Python\Interactive_Report_oo+merge.py"
echo %script%
call %base%\Scripts\activate.bat %base%
echo %*%
call python.exe -- %script% %*% -dp 1000 -l -s
REM pause