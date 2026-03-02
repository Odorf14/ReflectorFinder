@echo off
:: Absolute paths to venv, DLL, and script
set "PY64=C:\Users\nucamendi.r\OneDrive - Elettric 80\Documentos\_GitProjects\ReflectorFinder\venv64\Scripts\python.exe"
set "DLL64=C:\Users\nucamendi.r\OneDrive - Elettric 80\Documentos\_GitProjects\ReflectorFinder\Logger\TcAdsDllx64\TcAdsDll.dll"
set "SCRIPT=C:\Users\nucamendi.r\OneDrive - Elettric 80\Documentos\_GitProjects\ReflectorFinder\Logger\Logger.py"



:: Build the 64-bit exe
"%PY64%" -m PyInstaller --onefile --add-data "%DLL64%;." --name "ReflectorFinderLogger" "%SCRIPT%"

echo.
echo 64-bit build finished! Check the "dist" folder for ReflectorFinderLogger.exe
pause