@echo off
setlocal
cd /d "%~dp0"

python slai_windows_app.py --backend nn --nn-model-dir "artifacts/slai_nn" --nn-device auto --voice-input --voice-output --handsfree --wake-word "alice" %*
