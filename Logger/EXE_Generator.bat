@echo off

pyinstaller --onefile --add-data "TcAdsDll.dll;." --name "ReflectorFinder" Logger.py

echo.

echo Build finished! Check the "dist" folder for ReflectorFinder.exe

pause