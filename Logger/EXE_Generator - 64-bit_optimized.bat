@echo off
set "VENV=C:\Users\nucamendi.r\OneDrive - Elettric 80\Documentos\_GitProjects\ReflectorFinder\Logger\venv_build"
set "DLL64=C:\Users\nucamendi.r\OneDrive - Elettric 80\Documentos\_GitProjects\ReflectorFinder\Logger\TcAdsDllx64\TcAdsDll.dll"

"%VENV%\Scripts\pyinstaller.exe" ^
  --onefile ^
  --noconfirm ^
  --add-data "%DLL64%;." ^
  --name "ReflectorFinderLogger" ^
  --exclude-module PyQt5 ^
  --exclude-module PyQt6 ^
  --exclude-module matplotlib ^
  --exclude-module numpy ^
  --exclude-module scipy ^
  --exclude-module sklearn ^
  --exclude-module tkinter ^
  --exclude-module _tkinter ^
  Logger.py

echo.
echo Build finished! Check dist\ReflectorFinderLogger.exe
pause
