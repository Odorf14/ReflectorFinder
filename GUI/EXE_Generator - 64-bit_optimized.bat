@echo off
set "VENV=C:\Users\nucamendi.r\OneDrive - Elettric 80\Documentos\_GitProjects\ReflectorFinder\GUI\venv_build"
set "DLL64=C:\Users\nucamendi.r\OneDrive - Elettric 80\Documentos\_GitProjects\ReflectorFinder\GUI\TcAdsDllx64\TcAdsDll.dll"
set "ICON_PATH=C:\Users\nucamendi.r\OneDrive - Elettric 80\Documentos\_GitProjects\ReflectorFinder\GUI\Icon.ico"

"%VENV%\Scripts\pyinstaller.exe" ^
  --onedir ^
  --noconsole ^
  --noconfirm ^
  --icon="%ICON_PATH%" ^
  --add-data "%DLL64%;." ^
  --add-data "%ICON_PATH%;." ^
  --name "ReflectorFinderUI" ^
  --exclude-module PyQt6 ^
  --exclude-module matplotlib ^
  --exclude-module tkinter ^
  --exclude-module _tkinter ^
  GUI.py

echo.
echo Build finished! Check dist\ReflectorFinderUI\
pause