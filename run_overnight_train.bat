@echo off
setlocal

cd /d "%~dp0"

echo [SLAI] Starting overnight training...
echo [SLAI] Working dir: %CD%

if "%HF_TOKEN%"=="" (
  echo [SLAI] Warning: HF_TOKEN is not set in this terminal. Downloads may be slower.
)

set OUTPUT_DIR=artifacts\slai_nn_overnight
set LOG_DIR=artifacts\logs
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

set OUT_LOG=%LOG_DIR%\overnight_train.out.log
set ERR_LOG=%LOG_DIR%\overnight_train.err.log

echo [SLAI] Output dir: %OUTPUT_DIR%
echo [SLAI] Logs: %OUT_LOG% and %ERR_LOG%

python -u train_slai_nn.py ^
  --output-dir "%OUTPUT_DIR%" ^
  --profile 2b_like ^
  --device auto ^
  --max-per-dataset 1200 ^
  --max-local-rows 7000 ^
  --epochs 4 ^
  --resume 1>>"%OUT_LOG%" 2>>"%ERR_LOG%"

echo [SLAI] Training finished. Check:
echo   %OUTPUT_DIR%\train_summary.json
echo   %OUT_LOG%
echo   %ERR_LOG%

endlocal
