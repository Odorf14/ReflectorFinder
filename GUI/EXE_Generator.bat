@echo off

pyinstaller --onefile --noconsole --name "ReflectorFinderUI" GUI.py

echo.

echo Build finished! Check the "dist" folder for ReflectorFinderUI.exe

pause