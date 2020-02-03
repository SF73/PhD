@echo off
set base=%localappdata%\Continuum\miniconda3
REM echo %base%
set script="C:\Users\sylvain.finot\Cloud CNRS\Python\phu2dat.py"
call %base%\Scripts\activate.bat %base%
echo "%~dpn1%_2dat.dat"
call python.exe -- %script% %1 "%~dpn1%_2dat.dat"
pause