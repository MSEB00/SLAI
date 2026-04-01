# SLAI Windows App

## Run in Python

```powershell
python slai_windows_app.py --backend nn --nn-model-dir artifacts/slai_nn --nn-device auto
```

Or use the launcher:

```powershell
run_windows_app.bat
```

Quick launch choice:

- Use `run_windows_app.bat` while developing from source.
- Use `dist\SLAI-Desktop.exe` for direct app launch (no terminal required).

Voice-first defaults are enabled in the app:

- Voice input: on
- Voice output: on
- Wake word: `alice`
- Handsfree listening: on
- SLAI speaks a startup greeting when the app opens.
- Voice orb UI reacts in real time: `Listening`, `Thinking`, `Speaking`, `Idle`.

Wake usage:

- Say `alice` first, then your command (example: `alice set a reminder for 7 pm`).
- Built-in aliases also work: `hey alice`, `ok alice`, `okay alice`.

Use explicit flags only if you want to override defaults:

## Local SLAI NN Backend

Train local NN artifacts first:

```powershell
python train_slai_nn.py --output-dir artifacts/slai_nn --device auto
```

Then run the desktop app with local backend:

```powershell
python slai_windows_app.py --backend nn --nn-model-dir artifacts/slai_nn --nn-device auto
```

For lower RAM pressure:

```powershell
run_windows_app.bat --low-resource-mode
```

## Build `.exe`

```powershell
build_windows_app.bat
```

After build:

- Executable path: `dist\SLAI-Desktop.exe`
- The `.exe` can be large when Whisper/Torch dependencies are included.

## Optional startup flags

- `--backend nn`
- `--nn-model-dir artifacts/slai_nn`
- `--nn-device auto|cpu|cuda`
- `--low-resource-mode`
- `--voice-input`
- `--no-voice-input`
- `--voice-output`
- `--no-voice-output`
- `--stt-engine whisper|google`
- `--whisper-model tiny|base|small|...`
- `--disable-wake-word`
- `--handsfree` / `--no-handsfree`
- `--enable-autonomy`
