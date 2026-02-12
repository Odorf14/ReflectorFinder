@echo off

pyinstaller --onefile --noconsole --name "ReflectorFinderUI" --exclude-module PyQt6 GUI.py

echo.

echo Build finished! Check the "dist" folder for ReflectorFinderUI.exe

pause