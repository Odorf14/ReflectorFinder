@echo off

pyinstaller --onefile --add-data "TcAdsDll.dll;." --name "ReflectorFinderLogger" Logger.py

echo.

echo Build finished! Check the "dist" folder for ReflectorFinderLogger.exe

pause