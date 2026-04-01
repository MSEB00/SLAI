@echo off
setlocal
cd /d "%~dp0"

echo [SLAI] Installing PyInstaller...
python -m pip install pyinstaller
if errorlevel 1 goto :error

echo [SLAI] Building Windows app executable...
pyinstaller --noconfirm --clean --onefile --windowed --name "SLAI-Desktop" ^
  --hidden-import=tkinter --hidden-import=_tkinter --collect-submodules=tkinter ^
  slai_windows_app.py
if errorlevel 1 goto :error

echo [SLAI] Build complete.
echo [SLAI] Executable: dist\SLAI-Desktop.exe
exit /b 0

:error
echo [SLAI] Build failed.
exit /b 1
