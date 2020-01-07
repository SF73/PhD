@echo off
set base=%localappdata%\Continuum\miniconda3
REM echo %base%
set script="C:\Users\sylvain.finot\Cloud CNRS\Python\PlotSpectrum.py"
echo %script%
call %base%\Scripts\activate.bat %base%
echo %*
call python.exe -- %script% %*
pause