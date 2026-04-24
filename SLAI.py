import argparse
import json
import math
import queue
import re
import sys
import threading
import time
from difflib import SequenceMatcher
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import memory

MAX_TURNS = 12
active_model = None
ENABLE_SELF_REVIEW = True
FEEDBACK_LOG_FILE = Path("slai_feedback_log.jsonl")
SELF_LEARNING_FILE = Path("self_learning_memory.jsonl")
REMINDER_FILE = Path("reminders.json")
COGNITIVE_STATE_FILE = Path("cognitive_state.json")
HYBRID_BRAIN_STATE_FILE = Path("hybrid_brain_state.json")
DEFAULT_TIMEZONE = "Asia/Kolkata"
RUNTIME_BACKEND = "nn"
LOCAL_ENGINE = None
ENABLE_COGNITIVE_LOOP = True
ENABLE_HYBRID_BRAIN = True
ENABLE_AUTONOMOUS_CYCLES = True
AUTONOMOUS_MIN_TICK_SECONDS = 5
AUTONOMOUS_GOAL_NUDGE_MINUTES = 15
LOW_RESOURCE_MODE = False
SELF_LEARNING_MAX_ROWS = 5000
FEEDBACK_LOG_MAX_ROWS = 5000
HYBRID_EPISODIC_MAX_ROWS = 300

STATE_LOCK = threading.RLock()
PRINT_LOCK = threading.Lock()

VOICE_INPUT_ENGINE = None
VOICE_OUTPUT_ENGINE = None
VOICE_INPUT_ENABLED = False
VOICE_OUTPUT_ENABLED = False
VOICE_INPUT_BACKEND = "google"
WAKE_WORD_ENABLED = False
WAKE_WORD_PHRASE = "alice"
WAKE_WORD_ALIASES = ["alice", "hey alice", "ok alice", "okay alice"]
AUTONOMOUS_THREAD = None
AUTONOMOUS_STOP_EVENT = None
AUTONOMOUS_INTERVAL_SECONDS = 20

TIMEZONE_ALIASES = {
    "IST": "Asia/Kolkata",
    "UTC": "UTC",
    "GMT": "Etc/GMT",
    "EDT": "America/New_York",
    "EST": "America/New_York",
    "CDT": "America/Chicago",
    "CST": "America/Chicago",
    "MDT": "America/Denver",
    "MST": "America/Denver",
    "PDT": "America/Los_Angeles",
    "PST": "America/Los_Angeles",
}

TIMEZONE_DISPLAY = {
    "Asia/Kolkata": "IST",
    "UTC": "UTC",
    "Etc/GMT": "GMT",
    "America/New_York": "ET",
    "America/Chicago": "CT",
    "America/Denver": "MT",
    "America/Los_Angeles": "PT",
}

MEMORY_EXTRACTOR_PROMPT = """
Extract at most one stable user fact from the user's message.

Return JSON only (no extra text) using this schema:
{"store": boolean, "key": string, "value": string, "evidence": string}

Rules:
- If there is no storable user fact, return {"store": false}.
- Use a descriptive snake_case key.
- Store only clear user facts (identity, preferences, profile details, devices, goals).
- "evidence" must be an exact quote copied from the user message that supports the value.
- Keep value concise.
""".strip()

SELF_REVIEW_PROMPT = """
You are SLAI's internal verifier.

Task:
Check the draft assistant reply for:
- Hallucinations or unsupported claims
- Contradictions with known facts
- Overconfidence when uncertain
- Missing direct answer to the user question

Return JSON only:
{
  "pass": boolean,
  "revised_reply": string,
  "issues": [string],
  "confidence": number
}

Rules:
- If draft is safe and useful, set pass=true and revised_reply to the original draft.
- If draft has issues, set pass=false and provide a corrected revised_reply.
- Do not mention this verification process in the revised reply.
- Keep tone natural and concise.
""".strip()

REMINDER_EXTRACTOR_PROMPT = """
You extract reminder requests from user text.

Return JSON only:
{
  "intent": "create|list|none",
  "task": string,
  "when": string
}

Rules:
- intent=create only if the user clearly asks for a reminder.
- task should be concise and preserve user meaning.
- when should capture the user's time phrase exactly (e.g., "tomorrow 5 pm", "in 20 minutes").
- intent=list if user asks to show/list/remindings/reminders.
- If unclear, use intent=none.
""".strip()

COGNITIVE_PLANNER_PROMPT = """
You are SLAI's cognitive planner.

Analyze the latest user message in context and return JSON only:
{
  "intent": string,
  "store_goal": boolean,
  "goal_title": string,
  "next_actions": [string],
  "risk_flags": [string]
}

Rules:
- intent should be concise (e.g., "schedule", "question", "project_planning", "reflection", "task_execution").
- store_goal=true only if user clearly states a long-term objective.
- goal_title should be short and concrete when store_goal=true, else empty string.
- next_actions should be practical steps SLAI can take in conversation.
- risk_flags includes uncertainty or safety concerns; use empty list when none.
- Return valid JSON only.
""".strip()

VALID_MEMORY_KEY = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
GENERIC_MEMORY_KEYS = {"key", "value", "fact", "memory", "field", "data", "user_fact"}
LEGACY_VALUE_PATTERN = re.compile(r"^\s*([^,=]+)\s*,\s*value\s*=\s*(.+)\s*$", re.IGNORECASE)


class TextToSpeechEngine:
    def __init__(self, rate=180):
        import pyttsx3

        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", rate)
        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._speaking_event = threading.Event()
        self._worker = threading.Thread(target=self._run, name="slai-tts", daemon=True)
        self._worker.start()

    def _run(self):
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is None:
                break

            text = str(item).strip()
            if not text:
                continue

            try:
                self._speaking_event.set()
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception:
                pass
            finally:
                self._speaking_event.clear()

    def speak(self, text):
        if text is None:
            return
        self._queue.put(str(text))

    def is_speaking(self):
        if self._stop_event.is_set():
            return False
        return self._speaking_event.is_set() or (not self._queue.empty())

    def shutdown(self):
        self._stop_event.set()
        self._queue.put(None)
        if self._worker.is_alive():
            self._worker.join(timeout=2)


class SpeechToTextEngine:
    def __init__(
        self,
        adjust_noise_duration=0.5,
        engine="google",
        whisper_model_name="base",
        whisper_language=None,
        whisper_device="auto",
    ):
        import speech_recognition as sr

        self.sr = sr
        self.recognizer = sr.Recognizer()
        self.adjust_noise_duration = adjust_noise_duration
        self.engine = str(engine).strip().lower()
        self.whisper_model_name = whisper_model_name
        self.whisper_language = whisper_language
        self.whisper_device = whisper_device
        self.whisper_model = None
        self._np = None

        if self.engine == "whisper":
            try:
                import numpy as np
                import torch
                import whisper
            except Exception as exc:
                raise RuntimeError(f"Whisper dependencies unavailable: {exc}")

            if whisper_device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = whisper_device

            self._np = np
            self.whisper_device = device
            self.whisper_model = whisper.load_model(whisper_model_name, device=device)

    def _estimate_audio_level(self, audio):
        try:
            pcm_bytes = audio.get_raw_data(convert_rate=16000, convert_width=2)
            if not pcm_bytes:
                return 0.0
            samples = memoryview(pcm_bytes).cast("h")
            if not samples:
                return 0.0

            energy = 0.0
            for sample in samples:
                energy += float(sample) * float(sample)
            rms = math.sqrt(energy / float(len(samples))) / 32768.0
            normalized = min(1.0, max(0.0, rms * 8.0))
            return normalized
        except Exception:
            return 0.0

    def listen_once_with_level(self, timeout=5, phrase_time_limit=20):
        try:
            with self.sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=self.adjust_noise_duration)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        except Exception as exc:
            return None, f"Voice input error: {exc}", 0.0

        audio_level = self._estimate_audio_level(audio)

        if self.engine == "whisper":
            try:
                pcm_bytes = audio.get_raw_data(convert_rate=16000, convert_width=2)
                audio_np = self._np.frombuffer(pcm_bytes, dtype=self._np.int16).astype(self._np.float32) / 32768.0
                result = self.whisper_model.transcribe(
                    audio_np,
                    language=self.whisper_language if self.whisper_language else None,
                    fp16=self.whisper_device == "cuda",
                )
                text = str(result.get("text", "")).strip()
                if not text:
                    return None, "I couldn't understand the voice input.", audio_level
                return text, None, audio_level
            except Exception as exc:
                return None, f"Whisper transcription failed: {exc}", audio_level

        try:
            text = self.recognizer.recognize_google(audio)
            return text.strip(), None, audio_level
        except self.sr.WaitTimeoutError:
            return None, "No speech detected in time.", audio_level
        except self.sr.UnknownValueError:
            return None, "I couldn't understand the voice input.", audio_level
        except self.sr.RequestError as exc:
            return None, f"Speech recognition service error: {exc}", audio_level
        except Exception as exc:
            return None, f"Speech recognition failed: {exc}", audio_level

    def listen_once(self, timeout=5, phrase_time_limit=20):
        text, error, _ = self.listen_once_with_level(timeout=timeout, phrase_time_limit=phrase_time_limit)
        return text, error


def parse_runtime_args():
    parser = argparse.ArgumentParser(description="Run SLAI with local NN backend.")
    parser.add_argument("--backend", choices=["nn"], default="nn")
    parser.add_argument("--nn-model-dir", default="artifacts/slai_nn", help="Path to local SLAI NN artifacts.")
    parser.add_argument("--nn-device", choices=["auto", "cpu", "cuda"], default="auto", help="Device for local NN backend.")
    parser.add_argument("--max_new_tokens", type=int, default=220)
    parser.add_argument("--low-resource-mode", action="store_true", help="Disable auxiliary LLM passes to reduce memory usage.")
    parser.add_argument("--voice-input", action="store_true", help="Enable microphone speech-to-text.")
    parser.add_argument("--voice-output", action="store_true", help="Enable text-to-speech for SLAI replies.")
    parser.add_argument("--stt-engine", choices=["google", "whisper"], default="whisper", help="Speech-to-text backend.")
    parser.add_argument("--whisper-model", default="base", help="Whisper model name (tiny, base, small, ...).")
    parser.add_argument("--whisper-language", default="", help="Optional language code for Whisper (e.g. en, hi).")
    parser.add_argument("--whisper-device", choices=["auto", "cpu", "cuda"], default="auto", help="Device for Whisper.")
    parser.add_argument("--wake-word", default="alice", help="Wake word phrase for voice mode.")
    parser.add_argument("--disable-wake-word", action="store_true", help="Disable wake-word gating in voice mode.")
    parser.add_argument("--stt-timeout", type=int, default=5, help="Microphone listen timeout in seconds.")
    parser.add_argument("--stt-phrase-time-limit", type=int, default=20, help="Max phrase duration in seconds.")
    parser.add_argument("--autonomous-interval", type=int, default=20, help="Autonomous cycle interval in seconds.")
    parser.add_argument("--disable-autonomy", action="store_true", help="Disable autonomous background cycles.")
    return parser.parse_args()


def initialize_runtime(args):
    global RUNTIME_BACKEND
    global LOCAL_ENGINE
    global active_model
    global LOW_RESOURCE_MODE

    LOW_RESOURCE_MODE = bool(getattr(args, "low_resource_mode", False))

    RUNTIME_BACKEND = args.backend
    LOCAL_ENGINE = None
    if args.backend == "nn":
        from slai_nn import SLAINNEngine

        LOCAL_ENGINE = SLAINNEngine(
            model_dir=args.nn_model_dir,
            max_new_tokens=args.max_new_tokens,
            device=args.nn_device,
        )
        active_model = f"slai-nn ({args.nn_model_dir})"
        LOW_RESOURCE_MODE = True
    else:
        active_model = None


def initialize_modalities(args):
    global VOICE_INPUT_ENGINE
    global VOICE_OUTPUT_ENGINE
    global VOICE_INPUT_ENABLED
    global VOICE_OUTPUT_ENABLED
    global VOICE_INPUT_BACKEND
    global WAKE_WORD_ENABLED
    global WAKE_WORD_PHRASE
    global WAKE_WORD_ALIASES
    global ENABLE_AUTONOMOUS_CYCLES
    global AUTONOMOUS_INTERVAL_SECONDS

    VOICE_INPUT_ENABLED = bool(args.voice_input)
    VOICE_OUTPUT_ENABLED = bool(args.voice_output)
    VOICE_INPUT_BACKEND = str(args.stt_engine).strip().lower()
    WAKE_WORD_PHRASE = normalize_wake_phrase(args.wake_word) or "alice"
    WAKE_WORD_ALIASES = build_wake_word_aliases(WAKE_WORD_PHRASE)
    WAKE_WORD_ENABLED = VOICE_INPUT_ENABLED and not bool(args.disable_wake_word)
    ENABLE_AUTONOMOUS_CYCLES = not bool(args.disable_autonomy)
    AUTONOMOUS_INTERVAL_SECONDS = max(AUTONOMOUS_MIN_TICK_SECONDS, int(args.autonomous_interval))

    if VOICE_OUTPUT_ENABLED:
        try:
            VOICE_OUTPUT_ENGINE = TextToSpeechEngine()
            print("[Voice] TTS enabled (pyttsx3).")
        except Exception as exc:
            VOICE_OUTPUT_ENGINE = None
            VOICE_OUTPUT_ENABLED = False
            print(f"[Voice] TTS unavailable: {exc}")

    if VOICE_INPUT_ENABLED:
        try:
            VOICE_INPUT_ENGINE = SpeechToTextEngine(
                engine=VOICE_INPUT_BACKEND,
                whisper_model_name=args.whisper_model,
                whisper_language=(args.whisper_language.strip() or None),
                whisper_device=args.whisper_device,
            )
            if VOICE_INPUT_BACKEND == "whisper":
                print(f"[Voice] STT enabled (whisper/{args.whisper_model}, device={VOICE_INPUT_ENGINE.whisper_device}).")
            else:
                print("[Voice] STT enabled (speech_recognition/google).")
            print("[Voice] Press Enter to talk by mic.")
        except Exception as exc:
            if VOICE_INPUT_BACKEND == "whisper":
                try:
                    VOICE_INPUT_ENGINE = SpeechToTextEngine(engine="google")
                    VOICE_INPUT_BACKEND = "google"
                    print(f"[Voice] Whisper unavailable, fell back to google STT: {exc}")
                    print("[Voice] STT enabled (speech_recognition/google).")
                    print("[Voice] Press Enter to talk by mic.")
                except Exception as fallback_exc:
                    VOICE_INPUT_ENGINE = None
                    VOICE_INPUT_ENABLED = False
                    print(f"[Voice] STT unavailable: {fallback_exc}")
            else:
                VOICE_INPUT_ENGINE = None
                VOICE_INPUT_ENABLED = False
                print(f"[Voice] STT unavailable: {exc}")


def configure_console_io():
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if not reconfigure:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except OSError:
            # Keep current encoding if the environment doesn't allow reconfigure.
            pass


def safe_print_reply(reply, speak=True):
    with PRINT_LOCK:
        try:
            print("SLAI:", reply, flush=True)
        except UnicodeEncodeError:
            print("SLAI:", reply.encode("ascii", errors="replace").decode("ascii"), flush=True)

    if speak and VOICE_OUTPUT_ENGINE is not None:
        VOICE_OUTPUT_ENGINE.speak(reply)


def is_wake_word_enabled():
    return bool(WAKE_WORD_ENABLED and VOICE_INPUT_ENABLED and VOICE_INPUT_ENGINE is not None)


def normalize_wake_phrase(value):
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def build_wake_word_aliases(primary_wake_word):
    candidates = [
        primary_wake_word,
        "alice",
        "alyce",
        "ellis",
        "alise",
        "hey alice",
        "ok alice",
        "okay alice",
        "slai",
        "hey slai",
        "ok slai",
        "okay slai",
    ]
    aliases = []
    for candidate in candidates:
        normalized = normalize_wake_phrase(candidate)
        if normalized and normalized not in aliases:
            aliases.append(normalized)
    return aliases


def _token_similarity(left, right):
    if left == right:
        return 1.0
    return SequenceMatcher(None, left, right).ratio()


def extract_after_wake_word(transcript, wake_word):
    normalized_text = normalize_wake_phrase(transcript)
    normalized_wake = normalize_wake_phrase(wake_word)
    if not normalized_text or not normalized_wake:
        return None

    text_tokens = re.findall(r"[a-z0-9]+", normalized_text)
    wake_tokens = re.findall(r"[a-z0-9]+", normalized_wake)
    if not wake_tokens or not text_tokens:
        return None

    # Check first few tokens with fuzzy matching to tolerate STT mistakes.
    for start in range(0, min(3, len(text_tokens))):
        end = start + len(wake_tokens)
        if end > len(text_tokens):
            break
        window = text_tokens[start:end]
        score = sum(_token_similarity(a, b) for a, b in zip(window, wake_tokens)) / float(len(wake_tokens))
        if score >= 0.74:
            remainder_tokens = text_tokens[end:]
            return " ".join(remainder_tokens).strip()

    return None


def extract_after_any_wake_word(transcript, wake_words):
    for phrase in wake_words or []:
        remainder = extract_after_wake_word(transcript, phrase)
        if remainder is not None:
            return remainder
    return None


def transcribe_from_microphone(stt_timeout, stt_phrase_time_limit):
    print("[Voice] Listening...", flush=True)
    transcript, error = VOICE_INPUT_ENGINE.listen_once(timeout=stt_timeout, phrase_time_limit=stt_phrase_time_limit)
    if error:
        safe_print_reply(error, speak=False)
        return ""

    transcript = transcript.strip()
    with PRINT_LOCK:
        print(f"You (voice): {transcript}", flush=True)
    return transcript


def capture_voice_with_wake_word(stt_timeout, stt_phrase_time_limit):
    transcript = transcribe_from_microphone(stt_timeout, stt_phrase_time_limit)
    if not transcript:
        return ""

    if not is_wake_word_enabled():
        return transcript

    wake_remainder = extract_after_any_wake_word(transcript, WAKE_WORD_ALIASES)
    if wake_remainder is None:
        safe_print_reply(f"Wake word not detected. Say '{WAKE_WORD_PHRASE}' first.", speak=False)
        return ""

    if wake_remainder:
        return wake_remainder

    safe_print_reply("Wake word detected. Now say your command.", speak=False)
    command = transcribe_from_microphone(stt_timeout, stt_phrase_time_limit)
    return command.strip()


def get_user_input(stt_timeout, stt_phrase_time_limit):
    if not VOICE_INPUT_ENABLED or VOICE_INPUT_ENGINE is None:
        return input("You: ").strip()

    typed = input("You (type or press Enter for mic): ").strip()
    if typed:
        return typed

    return capture_voice_with_wake_word(stt_timeout, stt_phrase_time_limit)


def parse_json_content(raw_text):
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        # Fallback when model accidentally includes extra text around JSON.
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(raw_text[start : end + 1])
        except json.JSONDecodeError:
            return None


def coerce_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    if isinstance(value, (int, float)):
        return value != 0
    return default


def resolve_timezone(raw_text):
    if raw_text is None:
        return None

    text = str(raw_text).strip()
    text = re.sub(r"[?.!,]+$", "", text).strip()
    if not text:
        return None

    alias_zone = TIMEZONE_ALIASES.get(text.upper())
    if alias_zone:
        return alias_zone

    candidates = [
        text,
        text.replace(" ", "_"),
        text.replace("-", "_"),
    ]
    for candidate in candidates:
        try:
            ZoneInfo(candidate)
            return candidate
        except ZoneInfoNotFoundError:
            continue
    return None


def get_timezone_display(zone_name):
    return TIMEZONE_DISPLAY.get(zone_name, zone_name)


def get_user_timezone():
    memory_data = memory.load_memory()
    for key in ("timezone_preference", "timezone"):
        value = memory_data.get(key)
        zone_name = resolve_timezone(value)
        if zone_name:
            return zone_name
    return DEFAULT_TIMEZONE


def set_user_timezone(zone_name):
    memory.add_fact("timezone_preference", zone_name)
    memory.add_fact("timezone", get_timezone_display(zone_name))


def now_in_user_timezone(zone_name=None):
    target_zone = zone_name or get_user_timezone()
    return datetime.now(timezone.utc).astimezone(ZoneInfo(target_zone))


def format_time(dt, zone_name):
    return f"{dt.strftime('%Y-%m-%d %H:%M:%S')} {get_timezone_display(zone_name)}"


def parse_iso_datetime(value):
    try:
        dt = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_reminders():
    with STATE_LOCK:
        if not REMINDER_FILE.exists():
            return []
        try:
            with REMINDER_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
        if not isinstance(data, list):
            return []

        reminders = []
        for item in data:
            if not isinstance(item, dict):
                continue
            if not item.get("task") or not item.get("due_utc"):
                continue
            if parse_iso_datetime(item.get("due_utc")) is None:
                continue
            reminders.append(item)
        return reminders


def save_reminders(reminders):
    with STATE_LOCK:
        temp_file = REMINDER_FILE.with_suffix(".tmp")
        with temp_file.open("w", encoding="utf-8") as f:
            json.dump(reminders, f, indent=2, ensure_ascii=False)
        temp_file.replace(REMINDER_FILE)


def add_reminder(task, due_local, zone_name, source_text):
    with STATE_LOCK:
        due_utc = due_local.astimezone(timezone.utc)
        reminder = {
            "id": f"r_{int(datetime.now(timezone.utc).timestamp() * 1000)}",
            "task": re.sub(r"\s+", " ", task.strip()),
            "due_utc": due_utc.isoformat(),
            "timezone": zone_name,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_text": source_text,
        }
        reminders = load_reminders()
        reminders.append(reminder)
        reminders.sort(key=lambda item: parse_iso_datetime(item["due_utc"]) or datetime.max.replace(tzinfo=timezone.utc))
        save_reminders(reminders)
        return reminder


def pop_due_reminders(now_utc):
    with STATE_LOCK:
        reminders = load_reminders()
        due = []
        upcoming = []
        for reminder in reminders:
            due_time = parse_iso_datetime(reminder.get("due_utc"))
            if due_time is None:
                continue
            if due_time <= now_utc:
                due.append(reminder)
            else:
                upcoming.append(reminder)
        if len(upcoming) != len(reminders):
            save_reminders(upcoming)
        return due


def format_reminder(reminder, zone_name):
    due_utc = parse_iso_datetime(reminder.get("due_utc"))
    if due_utc is None:
        return reminder.get("task", "Reminder")
    due_local = due_utc.astimezone(ZoneInfo(zone_name))
    return f"{reminder.get('task', 'Reminder')} at {format_time(due_local, zone_name)}"


def list_upcoming_reminders(zone_name, limit=5):
    reminders = load_reminders()
    lines = []
    for index, reminder in enumerate(reminders[:limit], start=1):
        reminder_id = reminder.get("id", "unknown")
        lines.append(f"- [{index}] ({reminder_id}) {format_reminder(reminder, zone_name)}")
    return lines


def normalize_selector(selector):
    text = re.sub(r"\s+", " ", str(selector or "").strip())
    text = text.strip("'\"")
    if not text:
        return ""
    numeric_match = re.fullmatch(r"(?:#|number\s+)?(\d+)", text, flags=re.IGNORECASE)
    if numeric_match:
        return numeric_match.group(1)
    return text


def select_reminder(reminders, selector):
    if not reminders:
        return None, None, "No reminders found."

    normalized = normalize_selector(selector)
    if not normalized:
        return None, None, "Please specify which reminder to target."

    id_match = re.fullmatch(r"(?:id\s+)?(r_\d+)", normalized, flags=re.IGNORECASE)
    if id_match:
        reminder_id = id_match.group(1).lower()
        for idx, reminder in enumerate(reminders):
            if str(reminder.get("id", "")).lower() == reminder_id:
                return idx, reminder, None
        return None, None, f"No reminder found for id '{reminder_id}'."

    if normalized.isdigit():
        index = int(normalized)
        if 1 <= index <= len(reminders):
            idx = index - 1
            return idx, reminders[idx], None
        return None, None, f"Reminder index {index} is out of range."

    query = normalized.lower()
    matches = []
    for idx, reminder in enumerate(reminders):
        task = str(reminder.get("task", "")).lower()
        source_text = str(reminder.get("source_text", "")).lower()
        if query in task or query in source_text:
            matches.append((idx, reminder))

    if not matches:
        return None, None, f"No reminder matched '{normalized}'."
    if len(matches) > 1:
        preview = ", ".join(str(item[0] + 1) for item in matches[:5])
        return None, None, f"Multiple reminders matched '{normalized}' (indexes: {preview}). Use id or index."
    return matches[0][0], matches[0][1], None


def delete_reminder(selector):
    with STATE_LOCK:
        reminders = load_reminders()
        idx, target, error = select_reminder(reminders, selector)
        if error:
            return None, error
        removed = reminders.pop(idx)
        save_reminders(reminders)
        return removed, None


def edit_reminder(selector, new_due_local=None, new_task=None, zone_name=None):
    with STATE_LOCK:
        reminders = load_reminders()
        idx, target, error = select_reminder(reminders, selector)
        if error:
            return None, error

        if new_task is not None:
            task = re.sub(r"\s+", " ", str(new_task).strip())
            if not task:
                return None, "Reminder task cannot be empty."
            target["task"] = task

        if new_due_local is not None:
            effective_zone = zone_name or get_user_timezone()
            target["due_utc"] = new_due_local.astimezone(timezone.utc).isoformat()
            target["timezone"] = effective_zone

        target["updated_utc"] = datetime.now(timezone.utc).isoformat()
        reminders[idx] = target
        reminders.sort(key=lambda item: parse_iso_datetime(item["due_utc"]) or datetime.max.replace(tzinfo=timezone.utc))
        save_reminders(reminders)
        return target, None


def parse_clock_text(clock_text):
    text = re.sub(r"\s+", " ", clock_text.strip().lower().replace(".", ""))
    match = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", text)
    if not match:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2) or "0")
    suffix = match.group(3)

    if suffix:
        if hour < 1 or hour > 12 or minute > 59:
            return None
        if hour == 12:
            hour = 0
        if suffix == "pm":
            hour += 12
    else:
        if hour > 23 or minute > 59:
            return None
    return hour, minute


def parse_reminder_time_phrase(when_text, zone_name):
    local_now = now_in_user_timezone(zone_name)
    text = re.sub(r"\s+", " ", when_text.strip().lower())
    text = re.sub(r"[?.!,]+$", "", text)

    relative_match = re.fullmatch(
        r"(?:in\s+)?(\d+)\s*(minute|minutes|min|mins|hour|hours|hr|hrs|day|days)",
        text,
    )
    if relative_match:
        amount = int(relative_match.group(1))
        unit = relative_match.group(2)
        if "min" in unit:
            return local_now + timedelta(minutes=amount), None
        if "hour" in unit or unit in {"hr", "hrs"}:
            return local_now + timedelta(hours=amount), None
        return local_now + timedelta(days=amount), None

    for prefix, day_offset in (("tomorrow", 1), ("today", 0)):
        if text.startswith(prefix):
            remainder = text[len(prefix) :].strip()
            if remainder.startswith("at "):
                remainder = remainder[3:].strip()
            if remainder:
                parsed = parse_clock_text(remainder)
                if not parsed:
                    return None, f"I couldn't parse time '{when_text}'."
                hour, minute = parsed
            else:
                hour, minute = 9, 0

            target_date = (local_now + timedelta(days=day_offset)).date()
            due_local = datetime(
                target_date.year,
                target_date.month,
                target_date.day,
                hour,
                minute,
                tzinfo=ZoneInfo(zone_name),
            )
            if day_offset == 0 and due_local <= local_now:
                return None, "That time already passed today. Try a future time."
            return due_local, None

    date_match = re.fullmatch(r"(\d{4}-\d{2}-\d{2})(?:\s+(?:at\s+)?)?(.*)", text)
    if date_match:
        date_text = date_match.group(1)
        time_text = date_match.group(2).strip()
        try:
            date_value = datetime.strptime(date_text, "%Y-%m-%d").date()
        except ValueError:
            return None, f"I couldn't parse date '{date_text}'."

        if time_text:
            parsed = parse_clock_text(time_text)
            if not parsed:
                return None, f"I couldn't parse time '{time_text}'."
            hour, minute = parsed
        else:
            hour, minute = 9, 0

        due_local = datetime(
            date_value.year,
            date_value.month,
            date_value.day,
            hour,
            minute,
            tzinfo=ZoneInfo(zone_name),
        )
        if due_local <= local_now:
            return None, "That reminder time is in the past."
        return due_local, None

    parsed = parse_clock_text(text)
    if parsed:
        hour, minute = parsed
        due_local = local_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if due_local <= local_now:
            due_local += timedelta(days=1)
        return due_local, None

    return None, "I couldn't parse the reminder time. Try 'in 20 minutes', 'tomorrow 5 pm', or '2026-03-14 09:30'."


def extract_reminder_request(user_input):
    lower_text = user_input.lower()
    if re.search(r"\b(?:show|list|view)\b.*\b(?:reminders?|remainders?)\b", lower_text):
        return {"intent": "list", "task": "", "when": ""}

    regex_patterns = [
        r"^\s*(?:please\s+)?remind me to (?P<task>.+?)\s+(?:at|on|in)\s+(?P<when>.+)\s*$",
        r"^\s*(?:please\s+)?remind me to (?P<task>.+?)\s+(?P<when>(?:tomorrow|today)\b.+)\s*$",
        r"^\s*(?:please\s+)?remind me to (?P<task>.+?)\s+(?P<when>\d{4}-\d{2}-\d{2}(?:\s+.+)?)\s*$",
        r"^\s*set (?:a\s+)?reminder(?:\s+to)?\s+(?P<task>.+?)\s+(?:at|on|in)\s+(?P<when>.+)\s*$",
        r"^\s*(?:please\s+)?remind me in\s+(?P<when>.+?)\s+to\s+(?P<task>.+)\s*$",
    ]
    for pattern in regex_patterns:
        match = re.match(pattern, user_input, flags=re.IGNORECASE)
        if match:
            return {
                "intent": "create",
                "task": match.group("task").strip(),
                "when": match.group("when").strip(),
            }

    if "remind" not in lower_text and "reminder" not in lower_text:
        return None

    response = chat_with_fallback(
        messages=[
            {"role": "system", "content": REMINDER_EXTRACTOR_PROMPT},
            {"role": "user", "content": user_input},
        ],
        response_format="json",
    )
    payload = parse_json_content(response["message"]["content"].strip())
    if not isinstance(payload, dict):
        return None

    intent = str(payload.get("intent", "none")).strip().lower()
    task = str(payload.get("task", "")).strip()
    when = str(payload.get("when", "")).strip()
    if intent not in {"create", "list"}:
        return None
    if intent == "create" and (not task or not when):
        return None
    return {"intent": intent, "task": task, "when": when}


def extract_reminder_delete_request(user_input):
    pattern = re.compile(
        r"^\s*(?:delete|remove|cancel)\s+(?:the\s+)?(?:reminders?|remainders?)\s+(?P<selector>.+?)\s*$",
        re.IGNORECASE,
    )
    match = pattern.match(user_input)
    if not match:
        return None
    selector = normalize_selector(match.group("selector"))
    return selector or None


def extract_reminder_edit_request(user_input):
    task_patterns = [
        r"^\s*(?:edit|update|change)\s+(?:the\s+)?(?:reminders?|remainders?)\s+(?P<selector>.+?)\s+(?:task|text|title)\s+(?:to\s+)?(?P<task>.+?)\s*$",
        r"^\s*rename\s+(?:the\s+)?(?:reminders?|remainders?)\s+(?P<selector>.+?)\s+to\s+(?P<task>.+?)\s*$",
    ]
    for pattern in task_patterns:
        match = re.match(pattern, user_input, flags=re.IGNORECASE)
        if match:
            return {
                "selector": normalize_selector(match.group("selector")),
                "new_when": None,
                "new_task": match.group("task").strip(),
            }

    time_patterns = [
        r"^\s*(?:edit|update|change|reschedule)\s+(?:the\s+)?(?:reminders?|remainders?)\s+(?P<selector>.+?)\s+(?:to|for|at)\s+(?P<when>.+?)\s*$",
        r"^\s*(?:move)\s+(?:the\s+)?(?:reminders?|remainders?)\s+(?P<selector>.+?)\s+(?:to|for|at)\s+(?P<when>.+?)\s*$",
    ]
    for pattern in time_patterns:
        match = re.match(pattern, user_input, flags=re.IGNORECASE)
        if match:
            return {
                "selector": normalize_selector(match.group("selector")),
                "new_when": match.group("when").strip(),
                "new_task": None,
            }
    return None


def default_cognitive_state():
    return {
        "goals": [],
        "active_goal_id": None,
        "last_plan": {},
        "recent_thoughts": [],
        "autonomous": {},
    }


def load_cognitive_state():
    with STATE_LOCK:
        if not COGNITIVE_STATE_FILE.exists():
            return default_cognitive_state()

        try:
            with COGNITIVE_STATE_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return default_cognitive_state()

        if not isinstance(data, dict):
            return default_cognitive_state()

        goals = data.get("goals", [])
        if not isinstance(goals, list):
            goals = []
        normalized_goals = []
        for item in goals:
            if not isinstance(item, dict):
                continue
            goal_id = str(item.get("id", "")).strip()
            title = re.sub(r"\s+", " ", str(item.get("title", "")).strip())
            status = str(item.get("status", "active")).strip().lower()
            if not goal_id or not title:
                continue
            if status not in {"active", "completed", "paused"}:
                status = "active"
            normalized_goals.append(
                {
                    "id": goal_id,
                    "title": title[:180],
                    "status": status,
                    "source": str(item.get("source", "unknown")).strip() or "unknown",
                    "created_utc": str(item.get("created_utc", "")),
                    "updated_utc": str(item.get("updated_utc", "")),
                }
            )

        active_goal_id = data.get("active_goal_id")
        if active_goal_id and not any(g["id"] == active_goal_id for g in normalized_goals):
            active_goal_id = None

        recent_thoughts = data.get("recent_thoughts", [])
        if not isinstance(recent_thoughts, list):
            recent_thoughts = []
        recent_thoughts = [item for item in recent_thoughts if isinstance(item, dict)][-30:]

        last_plan = data.get("last_plan", {})
        if not isinstance(last_plan, dict):
            last_plan = {}

        autonomous = data.get("autonomous", {})
        if not isinstance(autonomous, dict):
            autonomous = {}

        return {
            "goals": normalized_goals,
            "active_goal_id": active_goal_id,
            "last_plan": last_plan,
            "recent_thoughts": recent_thoughts,
            "autonomous": autonomous,
        }


def save_cognitive_state(state):
    with STATE_LOCK:
        temp_file = COGNITIVE_STATE_FILE.with_suffix(".tmp")
        with temp_file.open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        temp_file.replace(COGNITIVE_STATE_FILE)


def normalize_goal_selector(selector):
    return normalize_selector(selector)


def select_goal(state, selector):
    goals = state.get("goals", [])
    if not goals:
        return None, None, "No goals found."

    normalized = normalize_goal_selector(selector)
    if not normalized:
        return None, None, "Please specify which goal to target."

    id_match = re.fullmatch(r"(?:id\s+)?(g_\d+)", normalized, flags=re.IGNORECASE)
    if id_match:
        target_id = id_match.group(1).lower()
        for idx, goal in enumerate(goals):
            if goal["id"].lower() == target_id:
                return idx, goal, None
        return None, None, f"No goal found for id '{target_id}'."

    if normalized.isdigit():
        index = int(normalized)
        if 1 <= index <= len(goals):
            idx = index - 1
            return idx, goals[idx], None
        return None, None, f"Goal index {index} is out of range."

    query = normalized.lower()
    matches = [(idx, goal) for idx, goal in enumerate(goals) if query in goal["title"].lower()]
    if not matches:
        return None, None, f"No goal matched '{normalized}'."
    if len(matches) > 1:
        preview = ", ".join(str(item[0] + 1) for item in matches[:5])
        return None, None, f"Multiple goals matched '{normalized}' (indexes: {preview}). Use id or index."
    return matches[0][0], matches[0][1], None


def add_goal(title, source="user"):
    goal_title = re.sub(r"\s+", " ", str(title).strip())
    if not goal_title:
        return None, "Goal title cannot be empty."
    goal_title = goal_title[:180]

    state = load_cognitive_state()
    for goal in state["goals"]:
        if goal["title"].lower() == goal_title.lower() and goal["status"] != "completed":
            state["active_goal_id"] = goal["id"]
            goal["updated_utc"] = datetime.now(timezone.utc).isoformat()
            save_cognitive_state(state)
            return goal, None

    now = datetime.now(timezone.utc).isoformat()
    goal = {
        "id": f"g_{int(datetime.now(timezone.utc).timestamp() * 1000)}",
        "title": goal_title,
        "status": "active",
        "source": source,
        "created_utc": now,
        "updated_utc": now,
    }
    state["goals"].append(goal)
    state["active_goal_id"] = goal["id"]
    save_cognitive_state(state)
    return goal, None


def list_goals(limit=10):
    state = load_cognitive_state()
    goals = state.get("goals", [])
    if not goals:
        return []

    lines = []
    for index, goal in enumerate(goals[:limit], start=1):
        active_marker = " *active*" if goal["id"] == state.get("active_goal_id") else ""
        lines.append(f"- [{index}] ({goal['id']}) {goal['title']} [{goal['status']}] {active_marker}".rstrip())
    return lines


def complete_goal(selector):
    state = load_cognitive_state()
    idx, goal, error = select_goal(state, selector)
    if error:
        return None, error

    now = datetime.now(timezone.utc).isoformat()
    goal["status"] = "completed"
    goal["updated_utc"] = now
    state["goals"][idx] = goal
    if state.get("active_goal_id") == goal["id"]:
        replacement = next((g["id"] for g in state["goals"] if g["status"] == "active"), None)
        state["active_goal_id"] = replacement
    save_cognitive_state(state)
    return goal, None


def activate_goal(selector):
    state = load_cognitive_state()
    _, goal, error = select_goal(state, selector)
    if error:
        return None, error

    if goal["status"] == "completed":
        goal["status"] = "active"
        goal["updated_utc"] = datetime.now(timezone.utc).isoformat()
        for idx, item in enumerate(state["goals"]):
            if item["id"] == goal["id"]:
                state["goals"][idx] = goal
                break

    state["active_goal_id"] = goal["id"]
    save_cognitive_state(state)
    return goal, None


def extract_goal_add_request(user_input):
    patterns = [
        r"^\s*(?:add|create|set)\s+(?:a\s+)?goal(?:\s+to)?\s+(?P<title>.+?)\s*$",
        r"^\s*my goal is(?:\s+to)?\s+(?P<title>.+?)\s*$",
    ]
    for pattern in patterns:
        match = re.match(pattern, user_input, flags=re.IGNORECASE)
        if match:
            return match.group("title").strip()
    return None


def is_goal_list_query(user_input):
    return re.search(r"\b(?:show|list|view|what(?:'s| is))\b.*\bgoals?\b", user_input, flags=re.IGNORECASE) is not None


def extract_goal_complete_request(user_input):
    pattern = re.compile(r"^\s*(?:complete|finish|mark)\s+(?:the\s+)?goal\s+(?P<selector>.+?)\s*$", re.IGNORECASE)
    match = pattern.match(user_input)
    if not match:
        return None
    selector = normalize_goal_selector(match.group("selector"))
    return selector or None


def extract_goal_activate_request(user_input):
    patterns = [
        r"^\s*(?:focus|activate)\s+(?:on\s+)?(?:the\s+)?goal\s+(?P<selector>.+?)\s*$",
        r"^\s*set\s+active\s+goal\s+(?P<selector>.+?)\s*$",
    ]
    for pattern in patterns:
        match = re.match(pattern, user_input, flags=re.IGNORECASE)
        if match:
            selector = normalize_goal_selector(match.group("selector"))
            if selector:
                return selector
    return None


def build_goals_context():
    state = load_cognitive_state()
    goals = state.get("goals", [])
    if not goals:
        return "No long-term goals tracked yet."

    active_goal_id = state.get("active_goal_id")
    active_goal = next((goal for goal in goals if goal["id"] == active_goal_id), None)
    active_text = active_goal["title"] if active_goal else "None"

    open_goals = [goal for goal in goals if goal["status"] == "active"][:5]
    open_goal_text = "; ".join(goal["title"] for goal in open_goals) if open_goals else "None"
    return f"Active goal: {active_text}\nOpen goals: {open_goal_text}"


def infer_intent_heuristic(user_input):
    text = user_input.lower().strip()
    if "?" in user_input:
        return "question"
    if "remind" in text:
        return "schedule"
    if "goal" in text:
        return "goal_management"
    if any(keyword in text for keyword in ("plan", "build", "implement", "create")):
        return "project_planning"
    if any(keyword in text for keyword in ("hi", "hello", "bye")):
        return "social"
    return "general_request"


def run_cognitive_cycle(user_input, history, memory_context):
    if LOW_RESOURCE_MODE:
        return {
            "intent": infer_intent_heuristic(user_input),
            "store_goal": False,
            "goal_title": "",
            "next_actions": [],
            "risk_flags": ["low_resource_mode"],
        }

    state = load_cognitive_state()
    planner = {
        "intent": infer_intent_heuristic(user_input),
        "store_goal": False,
        "goal_title": "",
        "next_actions": [],
        "risk_flags": [],
    }

    try:
        planner_input = {
            "user_input": user_input,
            "memory_context": memory_context,
            "goals_context": build_goals_context(),
            "recent_history": build_history_snippet(history),
        }
        response = chat_with_fallback(
            messages=[
                {"role": "system", "content": COGNITIVE_PLANNER_PROMPT},
                {"role": "user", "content": json.dumps(planner_input, ensure_ascii=False)},
            ],
            response_format="json",
        )
        payload = parse_json_content(response["message"]["content"].strip())
        if isinstance(payload, dict):
            planner["intent"] = str(payload.get("intent", planner["intent"])).strip() or planner["intent"]
            planner["store_goal"] = coerce_bool(payload.get("store_goal", False), default=False)
            planner["goal_title"] = str(payload.get("goal_title", "")).strip()
            next_actions = payload.get("next_actions", [])
            if isinstance(next_actions, list):
                planner["next_actions"] = [str(item).strip() for item in next_actions if str(item).strip()][:5]
            risk_flags = payload.get("risk_flags", [])
            if isinstance(risk_flags, list):
                planner["risk_flags"] = [str(item).strip() for item in risk_flags if str(item).strip()][:5]
    except Exception:
        pass

    if planner["store_goal"] and planner["goal_title"]:
        add_goal(planner["goal_title"], source="planner")
        state = load_cognitive_state()

    state["last_plan"] = {
        "intent": planner["intent"],
        "next_actions": planner["next_actions"],
        "risk_flags": planner["risk_flags"],
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    state["recent_thoughts"].append(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "user_input": user_input[:240],
            "intent": planner["intent"],
            "next_actions": planner["next_actions"][:3],
            "risk_flags": planner["risk_flags"][:3],
        }
    )
    state["recent_thoughts"] = state["recent_thoughts"][-30:]
    save_cognitive_state(state)
    return planner


def build_cognitive_context():
    state = load_cognitive_state()
    goals_context = build_goals_context()
    last_plan = state.get("last_plan", {})
    intent = str(last_plan.get("intent", "unknown")).strip() or "unknown"
    actions = last_plan.get("next_actions", [])
    if not isinstance(actions, list):
        actions = []
    action_text = "; ".join(str(item) for item in actions[:3]) if actions else "None"
    risk_flags = last_plan.get("risk_flags", [])
    if not isinstance(risk_flags, list):
        risk_flags = []
    risk_text = "; ".join(str(item) for item in risk_flags[:3]) if risk_flags else "None"

    return (
        "Cognitive context:\n"
        f"- {goals_context}\n"
        f"- Inferred intent: {intent}\n"
        f"- Planned next actions: {action_text}\n"
        f"- Risk flags: {risk_text}\n"
        "- Use this context to stay goal-directed and consistent."
    )


def default_hybrid_brain_state():
    return {
        "focus": {"intent": "unknown", "active_goal": None, "updated_utc": ""},
        "open_tasks": [],
        "episodic_memory": [],
        "last_actions": [],
        "meta": {"version": "phase1"},
    }


def load_hybrid_brain_state():
    with STATE_LOCK:
        if not HYBRID_BRAIN_STATE_FILE.exists():
            return default_hybrid_brain_state()
        try:
            with HYBRID_BRAIN_STATE_FILE.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError):
            return default_hybrid_brain_state()
        if not isinstance(data, dict):
            return default_hybrid_brain_state()

        state = default_hybrid_brain_state()
        focus = data.get("focus", {})
        if isinstance(focus, dict):
            state["focus"] = {
                "intent": str(focus.get("intent", "unknown")).strip() or "unknown",
                "active_goal": focus.get("active_goal"),
                "updated_utc": str(focus.get("updated_utc", "")),
            }

        open_tasks = data.get("open_tasks", [])
        if isinstance(open_tasks, list):
            normalized = []
            for item in open_tasks:
                if not isinstance(item, dict):
                    continue
                title = re.sub(r"\s+", " ", str(item.get("title", "")).strip())
                if not title:
                    continue
                normalized.append(
                    {
                        "id": str(item.get("id", ""))[:40],
                        "title": title[:180],
                        "status": str(item.get("status", "open")).strip().lower() or "open",
                        "source": str(item.get("source", "hybrid")).strip() or "hybrid",
                        "updated_utc": str(item.get("updated_utc", "")),
                    }
                )
            state["open_tasks"] = normalized[:60]

        episodic = data.get("episodic_memory", [])
        if isinstance(episodic, list):
            state["episodic_memory"] = [item for item in episodic if isinstance(item, dict)][-HYBRID_EPISODIC_MAX_ROWS:]

        last_actions = data.get("last_actions", [])
        if isinstance(last_actions, list):
            state["last_actions"] = [str(item).strip() for item in last_actions if str(item).strip()][:10]
        return state


def save_hybrid_brain_state(state):
    with STATE_LOCK:
        temp_file = HYBRID_BRAIN_STATE_FILE.with_suffix(".tmp")
        with temp_file.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, ensure_ascii=False)
        temp_file.replace(HYBRID_BRAIN_STATE_FILE)


def _extract_task_hint(user_input):
    text = str(user_input or "").strip()
    lowered = text.lower()
    if not text:
        return None
    if any(kw in lowered for kw in ("todo", "to-do", "task", "remind me to", "plan for")):
        return text[:180]
    return None


def run_hybrid_brain_cycle(user_input, history, memory_context):
    if not ENABLE_HYBRID_BRAIN:
        return {}

    planner = run_cognitive_cycle(user_input, history, memory_context) if ENABLE_COGNITIVE_LOOP else {
        "intent": infer_intent_heuristic(user_input),
        "next_actions": [],
        "risk_flags": [],
    }

    state = load_hybrid_brain_state()
    now = datetime.now(timezone.utc).isoformat()
    cognitive_state = load_cognitive_state()
    active_goal_id = cognitive_state.get("active_goal_id")
    active_goal = None
    if active_goal_id:
        active_goal = next((item for item in cognitive_state.get("goals", []) if item.get("id") == active_goal_id), None)

    task_hint = _extract_task_hint(user_input)
    if task_hint:
        existing = next((task for task in state["open_tasks"] if task["title"].lower() == task_hint.lower()), None)
        if existing:
            existing["updated_utc"] = now
        else:
            state["open_tasks"].append(
                {
                    "id": f"t_{int(datetime.now(timezone.utc).timestamp() * 1000)}",
                    "title": task_hint,
                    "status": "open",
                    "source": "user",
                    "updated_utc": now,
                }
            )
            state["open_tasks"] = state["open_tasks"][-60:]

    state["focus"] = {
        "intent": str(planner.get("intent", "unknown")).strip() or "unknown",
        "active_goal": active_goal["title"] if active_goal else None,
        "updated_utc": now,
    }
    state["last_actions"] = [str(item).strip() for item in planner.get("next_actions", []) if str(item).strip()][:5]
    state["episodic_memory"].append(
        {
            "timestamp_utc": now,
            "user": str(user_input)[:240],
            "intent": state["focus"]["intent"],
            "risk_flags": [str(item).strip() for item in planner.get("risk_flags", []) if str(item).strip()][:3],
            "actions": state["last_actions"][:3],
        }
    )
    state["episodic_memory"] = state["episodic_memory"][-HYBRID_EPISODIC_MAX_ROWS:]
    save_hybrid_brain_state(state)
    return planner


def build_hybrid_brain_context():
    if not ENABLE_HYBRID_BRAIN:
        return ""

    state = load_hybrid_brain_state()
    focus = state.get("focus", {})
    intent = str(focus.get("intent", "unknown")).strip() or "unknown"
    active_goal = str(focus.get("active_goal") or "None").strip()
    open_tasks = state.get("open_tasks", [])
    open_count = len([task for task in open_tasks if str(task.get("status", "open")).lower() == "open"])

    recent = state.get("episodic_memory", [])[-3:]
    recent_lines = []
    for item in recent:
        user_text = str(item.get("user", "")).strip()
        item_intent = str(item.get("intent", "unknown")).strip()
        if user_text:
            recent_lines.append(f"- intent={item_intent}; user='{user_text[:120]}'")
    recent_text = "\n".join(recent_lines) if recent_lines else "- None"

    actions = state.get("last_actions", [])
    actions_text = "; ".join(actions[:3]) if actions else "None"
    return (
        "Hybrid brain context:\n"
        f"- Focus intent: {intent}\n"
        f"- Active goal: {active_goal}\n"
        f"- Open tasks: {open_count}\n"
        f"- Suggested actions: {actions_text}\n"
        "- Recent episodes:\n"
        f"{recent_text}\n"
        "- Stay consistent with this state, and do not invent hidden actions."
    )


def run_autonomous_goal_nudge(now_utc):
    if not ENABLE_COGNITIVE_LOOP:
        return None

    state = load_cognitive_state()
    goals = state.get("goals", [])
    active_goal_id = state.get("active_goal_id")
    active_goal = next((goal for goal in goals if goal["id"] == active_goal_id and goal["status"] == "active"), None)
    if active_goal is None:
        return None

    autonomous = state.get("autonomous", {})
    last_ping_raw = autonomous.get("last_goal_ping_utc")
    last_ping = parse_iso_datetime(last_ping_raw) if last_ping_raw else None
    min_interval = timedelta(minutes=AUTONOMOUS_GOAL_NUDGE_MINUTES)
    if last_ping is not None and now_utc - last_ping < min_interval:
        return None

    last_plan = state.get("last_plan", {})
    next_actions = last_plan.get("next_actions", [])
    if not isinstance(next_actions, list):
        next_actions = []
    first_action = str(next_actions[0]).strip() if next_actions else ""
    if not first_action:
        first_action = "Take one small concrete step and report progress."

    autonomous["last_goal_ping_utc"] = now_utc.isoformat()
    state["autonomous"] = autonomous
    state["recent_thoughts"].append(
        {
            "timestamp_utc": now_utc.isoformat(),
            "user_input": "<autonomous_cycle>",
            "intent": "autonomous_goal_nudge",
            "next_actions": [first_action],
            "risk_flags": [],
        }
    )
    state["recent_thoughts"] = state["recent_thoughts"][-30:]
    save_cognitive_state(state)
    return f"Autonomous check: active goal '{active_goal['title']}'. Suggested next step: {first_action}"


def autonomous_background_loop(stop_event, interval_seconds):
    tick_seconds = max(AUTONOMOUS_MIN_TICK_SECONDS, int(interval_seconds))

    while not stop_event.is_set():
        if stop_event.wait(tick_seconds):
            break

        now_utc = datetime.now(timezone.utc)
        try:
            zone_name = get_user_timezone()
            due_reminders = pop_due_reminders(now_utc)
            for due in due_reminders:
                safe_print_reply(f"Reminder: {format_reminder(due, zone_name)}")

            nudge = run_autonomous_goal_nudge(now_utc)
            if nudge:
                safe_print_reply(nudge)
        except Exception as exc:
            with PRINT_LOCK:
                print(f"[Autonomy] background cycle error: {exc}", flush=True)


def start_autonomous_loop(interval_seconds):
    global AUTONOMOUS_THREAD
    global AUTONOMOUS_STOP_EVENT

    if not ENABLE_AUTONOMOUS_CYCLES:
        return

    if AUTONOMOUS_THREAD is not None and AUTONOMOUS_THREAD.is_alive():
        return

    AUTONOMOUS_STOP_EVENT = threading.Event()
    AUTONOMOUS_THREAD = threading.Thread(
        target=autonomous_background_loop,
        args=(AUTONOMOUS_STOP_EVENT, interval_seconds),
        name="slai-autonomy",
        daemon=True,
    )
    AUTONOMOUS_THREAD.start()


def stop_autonomous_loop():
    global AUTONOMOUS_THREAD
    global AUTONOMOUS_STOP_EVENT

    if AUTONOMOUS_STOP_EVENT is not None:
        AUTONOMOUS_STOP_EVENT.set()
    if AUTONOMOUS_THREAD is not None and AUTONOMOUS_THREAD.is_alive():
        AUTONOMOUS_THREAD.join(timeout=2)
    AUTONOMOUS_THREAD = None
    AUTONOMOUS_STOP_EVENT = None


def shutdown_modalities():
    stop_autonomous_loop()
    if VOICE_OUTPUT_ENGINE is not None:
        try:
            VOICE_OUTPUT_ENGINE.shutdown()
        except Exception:
            pass


def extract_timezone_change_request(user_input):
    match = re.search(
        r"\b(?:set|change|switch|update)\b.*\btimezone\b.*\b(?:to|as)\b\s+(.+)$",
        user_input,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    raw_candidate = match.group(1).strip()
    raw_candidate = re.sub(r"[?.!,]+$", "", raw_candidate).strip()
    if not raw_candidate:
        return None

    candidates = [raw_candidate]
    parts = raw_candidate.split()
    if parts:
        candidates.extend([parts[0], parts[-1]])

    for candidate in candidates:
        zone_name = resolve_timezone(candidate)
        if zone_name:
            return zone_name
    return None


def is_timezone_query(user_input):
    return re.search(
        r"\b(?:what(?:'s| is)|which|current)\b.*\btimezone\b",
        user_input,
        flags=re.IGNORECASE,
    ) is not None


def is_time_query(user_input):
    return re.search(
        r"\b(?:what(?:'s| is)\s+(?:the\s+)?time|current\s+time|time\s+now|tell me(?:\s+the)?\s+time|what time is it)\b",
        user_input,
        flags=re.IGNORECASE,
    ) is not None


def extract_autonomy_command(user_input):
    text = user_input.strip().lower()
    if re.fullmatch(r"(?:show\s+)?autonomy\s+status", text):
        return "status"
    if re.fullmatch(r"(?:pause|stop|disable)\s+autonomy", text):
        return "pause"
    if re.fullmatch(r"(?:resume|start|enable)\s+autonomy", text):
        return "resume"
    return None


def extract_wake_word_command(user_input):
    text = user_input.strip().lower()
    if re.fullmatch(r"(?:show\s+)?wake[\s-]?word\s+status", text):
        return "status"
    if re.fullmatch(r"(?:enable|start|resume)\s+wake[\s-]?word", text):
        return "enable"
    if re.fullmatch(r"(?:disable|stop|pause)\s+wake[\s-]?word", text):
        return "disable"
    return None


def extract_brain_command(user_input):
    text = user_input.strip().lower()
    if re.fullmatch(r"(?:show\s+)?brain\s+status", text):
        return "status"
    return None


def format_hybrid_brain_status():
    state = load_hybrid_brain_state()
    focus = state.get("focus", {})
    intent = str(focus.get("intent", "unknown")).strip() or "unknown"
    active_goal = str(focus.get("active_goal") or "None").strip()
    open_tasks = [item for item in state.get("open_tasks", []) if str(item.get("status", "open")).lower() == "open"]
    recent = state.get("episodic_memory", [])[-3:]

    lines = [
        f"Hybrid brain intent: {intent}",
        f"Active goal: {active_goal}",
        f"Open tasks: {len(open_tasks)}",
    ]
    if recent:
        lines.append("Recent episodes:")
        for item in recent:
            user_text = str(item.get("user", "")).strip()
            if user_text:
                lines.append(f"- {user_text[:100]}")
    return "\n".join(lines)


def handle_utility_request(user_input):
    global WAKE_WORD_ENABLED

    zone_name = get_user_timezone()

    autonomy_command = extract_autonomy_command(user_input)
    if autonomy_command == "status":
        running = AUTONOMOUS_THREAD is not None and AUTONOMOUS_THREAD.is_alive()
        if not ENABLE_AUTONOMOUS_CYCLES:
            status = "disabled"
        else:
            status = "running" if running else "stopped"
        return True, f"Autonomous loop is {status}."
    if autonomy_command == "pause":
        stop_autonomous_loop()
        return True, "Autonomous loop paused."
    if autonomy_command == "resume":
        if not ENABLE_AUTONOMOUS_CYCLES:
            return True, "Autonomous loop is disabled by runtime flag."
        start_autonomous_loop(interval_seconds=AUTONOMOUS_INTERVAL_SECONDS)
        return True, "Autonomous loop resumed."

    wake_cmd = extract_wake_word_command(user_input)
    if wake_cmd == "status":
        configured = "enabled" if WAKE_WORD_ENABLED else "disabled"
        effective = "enabled" if is_wake_word_enabled() else "disabled"
        note = "" if VOICE_INPUT_ENABLED else " Voice input is currently off."
        return True, f"Wake word config is {configured}; active state is {effective} ({WAKE_WORD_PHRASE}).{note}"
    if wake_cmd == "enable":
        WAKE_WORD_ENABLED = True
        if VOICE_INPUT_ENABLED:
            return True, f"Wake word enabled ({WAKE_WORD_PHRASE})."
        return True, f"Wake word enabled in config ({WAKE_WORD_PHRASE}). It will activate when voice input is enabled."
    if wake_cmd == "disable":
        WAKE_WORD_ENABLED = False
        return True, "Wake word disabled."

    brain_cmd = extract_brain_command(user_input)
    if brain_cmd == "status":
        return True, format_hybrid_brain_status()

    timezone_change = extract_timezone_change_request(user_input)
    if timezone_change:
        set_user_timezone(timezone_change)
        local_now = now_in_user_timezone(timezone_change)
        return True, f"Timezone updated to {get_timezone_display(timezone_change)} ({timezone_change}). Current time: {format_time(local_now, timezone_change)}."

    if is_timezone_query(user_input):
        return True, f"Your active timezone is {get_timezone_display(zone_name)} ({zone_name})."

    if is_time_query(user_input):
        local_now = now_in_user_timezone(zone_name)
        return True, f"It's {format_time(local_now, zone_name)}."

    goal_title = extract_goal_add_request(user_input)
    if goal_title:
        goal, error = add_goal(goal_title, source="user")
        if error:
            return True, error
        return True, f"Goal saved: ({goal['id']}) {goal['title']} [{goal['status']}]"

    if is_goal_list_query(user_input):
        goals = list_goals(limit=15)
        if not goals:
            return True, "You have no tracked goals yet."
        return True, "Tracked goals:\n" + "\n".join(goals)

    complete_selector = extract_goal_complete_request(user_input)
    if complete_selector:
        goal, error = complete_goal(complete_selector)
        if error:
            return True, error
        return True, f"Goal completed: ({goal['id']}) {goal['title']}"

    activate_selector = extract_goal_activate_request(user_input)
    if activate_selector:
        goal, error = activate_goal(activate_selector)
        if error:
            return True, error
        return True, f"Active goal set to: ({goal['id']}) {goal['title']}"

    delete_selector = extract_reminder_delete_request(user_input)
    if delete_selector:
        removed, error = delete_reminder(delete_selector)
        if error:
            return True, error
        return True, f"Reminder deleted: {format_reminder(removed, zone_name)}"

    edit_request = extract_reminder_edit_request(user_input)
    if edit_request:
        new_due_local = None
        if edit_request["new_when"]:
            new_due_local, error = parse_reminder_time_phrase(edit_request["new_when"], zone_name)
            if error:
                return True, error

        edited, error = edit_reminder(
            selector=edit_request["selector"],
            new_due_local=new_due_local,
            new_task=edit_request["new_task"],
            zone_name=zone_name,
        )
        if error:
            return True, error
        return True, f"Reminder updated: {format_reminder(edited, zone_name)}"

    reminder_request = extract_reminder_request(user_input)
    if reminder_request and reminder_request["intent"] == "list":
        upcoming = list_upcoming_reminders(zone_name, limit=10)
        if not upcoming:
            return True, "You have no upcoming reminders."
        return True, "Upcoming reminders:\n" + "\n".join(upcoming)

    if reminder_request and reminder_request["intent"] == "create":
        due_local, error = parse_reminder_time_phrase(reminder_request["when"], zone_name)
        if error:
            return True, error
        reminder = add_reminder(
            task=reminder_request["task"],
            due_local=due_local,
            zone_name=zone_name,
            source_text=user_input,
        )
        return True, f"Reminder set: {format_reminder(reminder, zone_name)}"

    return False, None


def normalize_learning_text(value, max_length=500):
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if not text:
        return ""
    return text[:max_length]


def trim_jsonl_file(path, max_rows):
    max_rows = max(1, int(max_rows))
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    if len(lines) <= max_rows:
        return
    trimmed = lines[-max_rows:]
    try:
        path.write_text("\n".join(trimmed) + "\n", encoding="utf-8")
    except OSError:
        return


def _tokenize_for_learning(value):
    return re.findall(r"[a-z0-9]+", str(value or "").lower())


def _looks_like_time_reply(reply_text):
    lower = str(reply_text or "").lower()
    if "it's " in lower and "ist" in lower:
        return True
    if re.search(r"\b\d{4}-\d{2}-\d{2}\b", lower):
        return True
    return False


def _is_time_question(prompt_text):
    lower = str(prompt_text or "").lower()
    return any(
        phrase in lower
        for phrase in (
            "what time",
            "current time",
            "time is it",
            "timezone",
            "time zone",
            "date and time",
        )
    )


def is_self_learning_pair_valid(prompt, reply, source="chat"):
    prompt_text = str(prompt or "").strip()
    reply_text = str(reply or "").strip()
    if not prompt_text or not reply_text:
        return False

    if len(prompt_text) < 3 or len(reply_text) < 3:
        return False

    if re.search(r"(.)\1{10,}", reply_text):
        return False

    lower_reply = reply_text.lower()
    if lower_reply.startswith("error:") or "failed to" in lower_reply:
        return False

    if _looks_like_time_reply(reply_text) and not _is_time_question(prompt_text):
        return False

    # Reject obviously garbled/meta spills.
    bad_phrases = [
        "hello chatgpt",
        "tool reception",
        "specific needs and budget",
    ]
    if any(phrase in lower_reply for phrase in bad_phrases):
        return False

    # Keep only examples with at least weak lexical relevance in free-form chat.
    if str(source).lower() == "chat":
        p_tokens = set(_tokenize_for_learning(prompt_text))
        r_tokens = set(_tokenize_for_learning(reply_text))
        if len(p_tokens) >= 3:
            overlap = len(p_tokens & r_tokens) / float(max(1, len(p_tokens | r_tokens)))
            if overlap < 0.03 and not _is_time_question(prompt_text):
                return False

    return True


def append_self_learning_example(user_input, final_reply, source="chat"):
    prompt = normalize_learning_text(user_input, max_length=500)
    reply = normalize_learning_text(final_reply, max_length=700)
    if not prompt or not reply:
        return
    if not is_self_learning_pair_valid(prompt, reply, source=source):
        return

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "instruction": prompt,
        "response": reply,
        "source": source,
    }
    with STATE_LOCK:
        with SELF_LEARNING_FILE.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_self_learning_examples(limit=800):
    if not SELF_LEARNING_FILE.exists():
        return []

    try:
        lines = SELF_LEARNING_FILE.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    rows = []
    for raw in reversed(lines):
        raw = raw.strip()
        if not raw:
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            continue
        instruction = normalize_learning_text(item.get("instruction", ""), max_length=500)
        response = normalize_learning_text(item.get("response", ""), max_length=700)
        if not instruction or not response:
            continue
        if not is_self_learning_pair_valid(instruction, response, source=item.get("source", "chat")):
            continue
        rows.append({"instruction": instruction, "response": response})
        if len(rows) >= limit:
            break
    return rows


def build_self_learning_context(user_input, top_k=3):
    query_tokens = set(re.findall(r"[a-z0-9]+", str(user_input or "").lower()))
    if not query_tokens:
        return ""

    scored = []
    for item in load_self_learning_examples():
        prompt = item["instruction"]
        prompt_tokens = set(re.findall(r"[a-z0-9]+", prompt.lower()))
        if not prompt_tokens:
            continue
        score = len(query_tokens & prompt_tokens) / float(len(query_tokens | prompt_tokens))
        if score > 0:
            scored.append((score, item))

    if not scored:
        return ""

    scored.sort(key=lambda row: row[0], reverse=True)
    lines = []
    for index, (_, item) in enumerate(scored[: max(1, int(top_k))], start=1):
        lines.append(f"- Example {index} user: {item['instruction']}")
        lines.append(f"  Example {index} assistant: {item['response']}")

    return "Self-learning recall examples:\n" + "\n".join(lines)


def build_runtime_clock_context():
    zone_name = get_user_timezone()
    local_now = now_in_user_timezone(zone_name)
    return (
        "Runtime clock context:\n"
        f"- Current local time: {format_time(local_now, zone_name)}\n"
        f"- Active timezone: {get_timezone_display(zone_name)} ({zone_name})\n"
        "- If asked for time or timezone, use this context and do not invent values."
    )


def load_system_prompt(
    memory_context,
    runtime_clock_context="",
    cognitive_context="",
    hybrid_context="",
    self_learning_context="",
):
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        prompt = f.read()
    prompt = prompt.replace("{memory_context}", memory_context)
    extras = []
    if runtime_clock_context:
        extras.append(runtime_clock_context)
    if cognitive_context:
        extras.append(cognitive_context)
    if hybrid_context:
        extras.append(hybrid_context)
    if self_learning_context:
        extras.append(self_learning_context)
    if extras:
        prompt = f"{prompt}\n\n" + "\n\n".join(extras)
    return prompt


def enable_low_resource_mode(reason):
    global LOW_RESOURCE_MODE
    if LOW_RESOURCE_MODE:
        return
    LOW_RESOURCE_MODE = True
    print(f"[Runtime] Low-resource mode enabled: {reason}")


def chat_with_fallback(messages, response_format=None):
    global active_model
    if LOCAL_ENGINE is None:
        raise RuntimeError("Local backend not initialized.")
    if active_model and not getattr(chat_with_fallback, "_local_announced", False):
        print(f"[Model] Active model: {active_model}")
        chat_with_fallback._local_announced = True
    return LOCAL_ENGINE.chat(messages, response_format=response_format)


def normalize_memory_key(raw_key):
    if raw_key is None:
        return None

    key = str(raw_key).strip().lower().replace("-", "_").replace(" ", "_")
    key = re.sub(r"[^a-z0-9_]", "", key)
    key = re.sub(r"_+", "_", key).strip("_")

    if not key or key in GENERIC_MEMORY_KEYS:
        return None
    if not VALID_MEMORY_KEY.fullmatch(key):
        return None
    return key


def normalize_memory_value(raw_value):
    if raw_value is None:
        return None

    value = re.sub(r"\s+", " ", str(raw_value).strip())
    if not value or len(value) > 200:
        return None
    return value


def recover_fact(raw_key, raw_value, raw_evidence):
    key = normalize_memory_key(raw_key)
    value = normalize_memory_value(raw_value)
    evidence = normalize_memory_value(raw_evidence) if raw_evidence is not None else None
    if key and value:
        return key, value, evidence or value

    if value:
        legacy_match = LEGACY_VALUE_PATTERN.match(value)
        if legacy_match:
            legacy_key = normalize_memory_key(legacy_match.group(1))
            legacy_value = normalize_memory_value(legacy_match.group(2))
            if legacy_key and legacy_value:
                return legacy_key, legacy_value, legacy_value

    return None


def is_grounded_fact(value, evidence, user_input):
    user_text = re.sub(r"\s+", " ", user_input.lower().strip())
    value_text = re.sub(r"\s+", " ", value.lower().strip())
    evidence_text = re.sub(r"\s+", " ", evidence.lower().strip())

    if evidence_text not in user_text:
        return False

    if value_text in user_text:
        return True
    if value_text in evidence_text:
        return True

    return False


def extract_memory_fact(user_input):
    if LOW_RESOURCE_MODE:
        return None

    response = chat_with_fallback(
        messages=[
            {"role": "system", "content": MEMORY_EXTRACTOR_PROMPT},
            {"role": "user", "content": user_input},
        ],
        response_format="json",
    )
    raw_json = response["message"]["content"].strip()

    payload = parse_json_content(raw_json)

    if not isinstance(payload, dict):
        return None

    store = coerce_bool(payload.get("store", False), default=False)
    if not store:
        return None

    recovered = recover_fact(payload.get("key"), payload.get("value"), payload.get("evidence"))
    if not recovered:
        return None

    key, value, evidence = recovered
    if not is_grounded_fact(value, evidence, user_input):
        return None

    return key, value


def build_history_snippet(history, max_messages=6):
    snippet = history[-max_messages:] if len(history) > max_messages else history
    return [{"role": item["role"], "content": item["content"]} for item in snippet]


def verify_reply(user_input, memory_context, history, draft_reply):
    if LOW_RESOURCE_MODE:
        return draft_reply, {"pass": True, "issues": ["low_resource_mode"], "confidence": 0.0}

    review_input = {
        "user_input": user_input,
        "known_facts": memory_context,
        "recent_history": build_history_snippet(history),
        "draft_reply": draft_reply,
    }

    response = chat_with_fallback(
        messages=[
            {"role": "system", "content": SELF_REVIEW_PROMPT},
            {"role": "user", "content": json.dumps(review_input, ensure_ascii=False)},
        ],
        response_format="json",
    )
    payload = parse_json_content(response["message"]["content"].strip())
    if not isinstance(payload, dict):
        return draft_reply, {"pass": True, "issues": ["review_parse_failed"], "confidence": 0.0}

    review_pass = coerce_bool(payload.get("pass"), default=True)
    revised_reply = str(payload.get("revised_reply", "")).strip()
    issues = payload.get("issues", [])
    if not isinstance(issues, list):
        issues = [str(issues)]

    confidence = payload.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0

    if not revised_reply:
        revised_reply = draft_reply

    final_reply = draft_reply if review_pass else revised_reply
    if not final_reply.strip():
        final_reply = draft_reply

    review_data = {
        "pass": review_pass,
        "issues": issues[:5],
        "confidence": max(0.0, min(1.0, confidence)),
    }
    return final_reply.strip(), review_data


def append_feedback_log(user_input, draft_reply, final_reply, review_data):
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "user_input": user_input,
        "draft_reply": draft_reply,
        "final_reply": final_reply,
        "review": review_data,
    }
    with STATE_LOCK:
        with FEEDBACK_LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        trim_jsonl_file(FEEDBACK_LOG_FILE, FEEDBACK_LOG_MAX_ROWS)


def trim_history(messages, max_turns):
    max_messages = max_turns * 2
    if len(messages) > max_messages:
        del messages[: len(messages) - max_messages]


def main():
    configure_console_io()
    args = parse_runtime_args()

    try:
        initialize_runtime(args)
        initialize_modalities(args)
    except Exception as exc:
        print(f"SLAI: Failed to initialize runtime: {exc}")
        return

    print("SLAI Assistant v0.9")
    print("Type 'exit' to quit\n")
    print("Backend: slai-nn")
    print("Model:", active_model)
    if ENABLE_SELF_REVIEW:
        print("Self-review: enabled")
    if ENABLE_COGNITIVE_LOOP:
        print("Cognitive loop: enabled")
    if ENABLE_AUTONOMOUS_CYCLES:
        print(f"Autonomous loop: enabled ({AUTONOMOUS_INTERVAL_SECONDS}s)")
    else:
        print("Autonomous loop: disabled")
    if VOICE_INPUT_ENABLED:
        print(f"Voice input: enabled ({VOICE_INPUT_BACKEND})")
        if is_wake_word_enabled():
            print(f"Wake word: enabled ({WAKE_WORD_PHRASE})")
        else:
            print("Wake word: disabled")
    if VOICE_OUTPUT_ENABLED:
        print("Voice output: enabled")
    if LOW_RESOURCE_MODE:
        print("Low-resource mode: enabled")

    history = []
    start_autonomous_loop(interval_seconds=AUTONOMOUS_INTERVAL_SECONDS)

    try:
        while True:
            if not ENABLE_AUTONOMOUS_CYCLES:
                zone_name = get_user_timezone()
                due_reminders = pop_due_reminders(datetime.now(timezone.utc))
                for due in due_reminders:
                    safe_print_reply(f"Reminder: {format_reminder(due, zone_name)}")

            user_input = get_user_input(args.stt_timeout, args.stt_phrase_time_limit)

            if user_input.lower() == "exit":
                break
            if not user_input:
                continue

            try:
                handled, utility_reply = handle_utility_request(user_input)
                if handled:
                    safe_print_reply(utility_reply)
                    history.append({"role": "user", "content": user_input})
                    history.append({"role": "assistant", "content": utility_reply})
                    append_self_learning_example(user_input, utility_reply, source="utility")
                    trim_history(history, MAX_TURNS)
                    continue

                memory_fact = extract_memory_fact(user_input)
                if memory_fact:
                    memory.add_fact(memory_fact[0], memory_fact[1])

                memory_context = memory.get_memory_context()
                if ENABLE_HYBRID_BRAIN:
                    run_hybrid_brain_cycle(user_input, history, memory_context)
                elif ENABLE_COGNITIVE_LOOP:
                    run_cognitive_cycle(user_input, history, memory_context)
                runtime_clock_context = build_runtime_clock_context()
                cognitive_context = build_cognitive_context() if ENABLE_COGNITIVE_LOOP else ""
                hybrid_context = build_hybrid_brain_context() if ENABLE_HYBRID_BRAIN else ""
                self_learning_context = build_self_learning_context(user_input)
                system_prompt = load_system_prompt(
                    memory_context,
                    runtime_clock_context=runtime_clock_context,
                    cognitive_context=cognitive_context,
                    hybrid_context=hybrid_context,
                    self_learning_context=self_learning_context,
                )

                response = chat_with_fallback(
                    messages=[{"role": "system", "content": system_prompt}, *history, {"role": "user", "content": user_input}],
                )
                draft_reply = response["message"]["content"].strip()
                reply = draft_reply

                review_data = None
                if ENABLE_SELF_REVIEW and draft_reply:
                    reply, review_data = verify_reply(user_input, memory_context, history, draft_reply)
                    append_feedback_log(user_input, draft_reply, reply, review_data)
            except Exception as exc:
                print(f"SLAI: I ran into an error: {exc}")
                continue

            safe_print_reply(reply)
            append_self_learning_example(user_input, reply, source="chat")

            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": reply})
            trim_history(history, MAX_TURNS)
    finally:
        shutdown_modalities()


if __name__ == "__main__":
    main()
