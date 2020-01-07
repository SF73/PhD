@echo off
set base=%localappdata%\Continuum\miniconda3
REM echo %base%
set script="C:\Users\sylvain.finot\Cloud CNRS\Python\Interactive_Report_oo.py"
echo %script%
call %base%\Scripts\activate.bat %base%
echo %1%
call ipython.exe -- %script% %1% -dp 10 -l -s
pause