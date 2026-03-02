@echo off
:: Absolute paths to venv, DLL, and script
set "PY32=C:\Users\nucamendi.r\OneDrive - Elettric 80\Documentos\_GitProjects\ReflectorFinder\venv32\Scripts\python.exe"
set "DLL32=C:\Users\nucamendi.r\OneDrive - Elettric 80\Documentos\_GitProjects\ReflectorFinder\Logger\TcAdsDllx32\TcAdsDll.dll"
set "SCRIPT=C:\Users\nucamendi.r\OneDrive - Elettric 80\Documentos\_GitProjects\ReflectorFinder\Logger\Logger.py"



:: Build the 32-bit exe
::"%PY32%" -m PyInstaller --onefile --add-data "%DLL32%;." --name "ReflectorFinderLogger" "%SCRIPT%"
"%PY32%" -m PyInstaller --onefile --add-data "%DLL32%;." --name "ReflectorFinderLogger_x32" "%SCRIPT%"

echo.
echo 32-bit build finished! Check the "dist" folder for ReflectorFinderLogger.exe
pause