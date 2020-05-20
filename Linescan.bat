@echo off
set base=%localappdata%\Continuum\miniconda3
REM echo %base%
set script="C:\Users\sylvain.finot\Cloud CNRS\Python\Interactive_Report_oo+merge_clean.py"
echo %script%
call %base%\Scripts\activate.bat %base%
echo %*%
call ipython.exe -i -- %script% %*% -dp 1000 -l -s
pause