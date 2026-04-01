import argparse
import math
import queue
import threading
import time
from argparse import Namespace
from datetime import datetime, timezone

import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

import SLAI


class SLAIRuntime:
    def __init__(self, app_args):
        self.app_args = app_args
        self.history = []
        self._init_runtime()

    def _build_slai_namespace(self):
        return Namespace(
            backend=self.app_args.backend,
            nn_model_dir=self.app_args.nn_model_dir,
            nn_device=self.app_args.nn_device,
            max_new_tokens=self.app_args.max_new_tokens,
            low_resource_mode=self.app_args.low_resource_mode,
            voice_input=self.app_args.voice_input,
            voice_output=self.app_args.voice_output,
            stt_engine=self.app_args.stt_engine,
            whisper_model=self.app_args.whisper_model,
            whisper_language=self.app_args.whisper_language,
            whisper_device=self.app_args.whisper_device,
            wake_word=self.app_args.wake_word,
            disable_wake_word=self.app_args.disable_wake_word,
            stt_timeout=self.app_args.stt_timeout,
            stt_phrase_time_limit=self.app_args.stt_phrase_time_limit,
            autonomous_interval=self.app_args.autonomous_interval,
            disable_autonomy=self.app_args.disable_autonomy,
        )

    def _init_runtime(self):
        SLAI.configure_console_io()
        slai_args = self._build_slai_namespace()
        SLAI.initialize_runtime(slai_args)
        SLAI.initialize_modalities(slai_args)
        if not slai_args.disable_autonomy:
            SLAI.start_autonomous_loop(interval_seconds=slai_args.autonomous_interval)

    def process_message(self, user_input):
        handled, utility_reply = SLAI.handle_utility_request(user_input)
        if handled:
            reply = utility_reply
            self._append_history(user_input, reply)
            SLAI.append_self_learning_example(user_input, reply, source="utility")
            return reply

        memory_fact = SLAI.extract_memory_fact(user_input)
        if memory_fact:
            SLAI.memory.add_fact(memory_fact[0], memory_fact[1])

        memory_context = SLAI.memory.get_memory_context()
        if SLAI.ENABLE_COGNITIVE_LOOP:
            SLAI.run_cognitive_cycle(user_input, self.history, memory_context)

        runtime_clock_context = SLAI.build_runtime_clock_context()
        cognitive_context = SLAI.build_cognitive_context() if SLAI.ENABLE_COGNITIVE_LOOP else ""
        self_learning_context = SLAI.build_self_learning_context(user_input)
        system_prompt = SLAI.load_system_prompt(
            memory_context,
            runtime_clock_context=runtime_clock_context,
            cognitive_context=cognitive_context,
            self_learning_context=self_learning_context,
        )

        response = SLAI.chat_with_fallback(
            messages=[
                {"role": "system", "content": system_prompt},
                *self.history,
                {"role": "user", "content": user_input},
            ]
        )
        draft_reply = response["message"]["content"].strip()
        reply = draft_reply

        if SLAI.ENABLE_SELF_REVIEW and draft_reply:
            reply, review_data = SLAI.verify_reply(user_input, memory_context, self.history, draft_reply)
            SLAI.append_feedback_log(user_input, draft_reply, reply, review_data)

        self._append_history(user_input, reply)
        SLAI.append_self_learning_example(user_input, reply, source="chat")
        return reply

    def poll_due_reminders(self):
        zone_name = SLAI.get_user_timezone()
        due = SLAI.pop_due_reminders(datetime.now(timezone.utc))
        return [f"Reminder: {SLAI.format_reminder(item, zone_name)}" for item in due]

    def _append_history(self, user_input, reply):
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": reply})
        SLAI.trim_history(self.history, SLAI.MAX_TURNS)

    def shutdown(self):
        SLAI.shutdown_modalities()


class SLAIWindowsApp(tk.Tk):
    def __init__(self, runtime):
        super().__init__()
        self.runtime = runtime
        self.result_queue = queue.Queue()
        self.busy = False
        self.pending_inputs = []
        self.voice_listener_stop = threading.Event()
        self.voice_listener_thread = None
        self.voice_anim_phase = 0.0
        self.voice_visual_state = "idle"
        self._listen_count = 0
        self._listen_count_lock = threading.Lock()
        self._voice_level = 0.0
        self._voice_level_lock = threading.Lock()

        self.title("SLAI Desktop")
        self.geometry("980x680")
        self.minsize(760, 520)

        self._build_ui()
        self._set_status("Ready")
        greeting = "Hello, I am SLAI."
        spoken_greeting = "Hello, I am SLAI."
        if SLAI.VOICE_INPUT_ENABLED:
            greeting += " Voice mode is active."
            spoken_greeting += " Voice mode is active."
            if SLAI.is_wake_word_enabled():
                greeting += f" Say '{SLAI.WAKE_WORD_PHRASE}' to wake me."
                spoken_greeting += " Say the wake phrase to wake me."
            else:
                greeting += " Speak directly to give commands."
                spoken_greeting += " Speak directly to give commands."
        else:
            greeting += " Type your message below."
            spoken_greeting += " Type your message below."
        self._append_chat("SLAI", greeting, speak=False)
        self._speak(spoken_greeting)

        self.after(150, self._poll_results)
        self.after(1000, self._poll_due_reminders)
        self.after(80, self._animate_voice_ui)
        # Delay hands-free listening so startup speech does not trigger wake-word capture.
        self.after(2000, self._start_voice_listener)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        root = ttk.Frame(self, padding=12)
        root.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(root)
        header.pack(fill=tk.X)

        self.backend_var = tk.StringVar(value=SLAI.RUNTIME_BACKEND)
        backend_label = ttk.Label(header, text=f"Backend: {self.backend_var.get()}")
        backend_label.pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="")
        status_label = ttk.Label(header, textvariable=self.status_var)
        status_label.pack(side=tk.RIGHT)

        voice_card = ttk.Frame(root)
        voice_card.pack(fill=tk.X, pady=(10, 8))

        voice_header = ttk.Frame(voice_card)
        voice_header.pack(fill=tk.X, pady=(0, 6))

        self.voice_state_var = tk.StringVar(value="Idle")
        voice_title = ttk.Label(voice_header, text="SLAI Voice", font=("Segoe UI", 11, "bold"))
        voice_title.pack(side=tk.LEFT)
        voice_state = ttk.Label(voice_header, textvariable=self.voice_state_var)
        voice_state.pack(side=tk.RIGHT)

        self.voice_canvas = tk.Canvas(
            voice_card,
            height=120,
            highlightthickness=0,
            bd=0,
            bg="#0F172A",
        )
        self.voice_canvas.pack(fill=tk.X)
        self.voice_backdrop = self.voice_canvas.create_rectangle(0, 0, 1, 1, fill="#0F172A", outline="")
        self.voice_ring = self.voice_canvas.create_oval(0, 0, 1, 1, outline="#38BDF8", width=3)
        self.voice_core = self.voice_canvas.create_oval(0, 0, 1, 1, fill="#22D3EE", outline="")
        self.voice_canvas.bind("<Configure>", self._on_voice_canvas_resize)

        self.chat_box = ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED, font=("Segoe UI", 11))
        self.chat_box.pack(fill=tk.BOTH, expand=True, pady=(10, 8))

        input_group = ttk.LabelFrame(root, text="Chat with SLAI")
        input_group.pack(fill=tk.X)

        input_row = ttk.Frame(input_group, padding=(8, 6))
        input_row.pack(fill=tk.X)

        self.input_var = tk.StringVar()
        self.input_entry = ttk.Entry(input_row, textvariable=self.input_var)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.input_entry.bind("<Return>", self._on_send_enter)

        self.send_btn = ttk.Button(input_row, text="Send", command=self._send_message)
        self.send_btn.pack(side=tk.LEFT, padx=(8, 0))

        self.voice_btn = ttk.Button(input_row, text="Mic", command=self._mic_message)
        self.voice_btn.pack(side=tk.LEFT, padx=(8, 0))

    def _append_chat(self, speaker, text, speak=False):
        timestamp = datetime.now().strftime("%H:%M:%S")
        message = f"[{timestamp}] {speaker}: {text}\n\n"

        self.chat_box.configure(state=tk.NORMAL)
        self.chat_box.insert(tk.END, message)
        self.chat_box.see(tk.END)
        self.chat_box.configure(state=tk.DISABLED)

        if speak:
            self._speak(text)

    def _speak(self, text):
        if SLAI.VOICE_OUTPUT_ENGINE is not None:
            SLAI.VOICE_OUTPUT_ENGINE.speak(str(text))

    def _mark_listening(self, active):
        with self._listen_count_lock:
            if active:
                self._listen_count += 1
            else:
                self._listen_count = max(0, self._listen_count - 1)

    def _set_voice_level(self, level):
        try:
            numeric = float(level)
        except (TypeError, ValueError):
            return
        clamped = max(0.0, min(1.0, numeric))
        with self._voice_level_lock:
            self._voice_level = max(self._voice_level, clamped)

    def _get_voice_level(self):
        with self._voice_level_lock:
            return self._voice_level

    def _decay_voice_level(self, decay):
        with self._voice_level_lock:
            self._voice_level = max(0.0, self._voice_level - decay)
            return self._voice_level

    def _is_listening(self):
        with self._listen_count_lock:
            return self._listen_count > 0

    def _is_speaking(self):
        engine = SLAI.VOICE_OUTPUT_ENGINE
        if engine is None:
            return False
        checker = getattr(engine, "is_speaking", None)
        if not callable(checker):
            return False
        try:
            return bool(checker())
        except Exception:
            return False

    def _get_visual_state(self):
        if self._is_speaking():
            return "speaking"
        if self.busy:
            return "thinking"
        if self._is_listening():
            return "listening"
        if SLAI.VOICE_INPUT_ENABLED and self.runtime.app_args.handsfree:
            return "standby"
        return "idle"

    def _state_to_palette(self, state):
        if state == "speaking":
            return ("Speaking", "#34D399", "#10B981", "#064E3B", 1.16, 0.20)
        if state == "thinking":
            return ("Thinking", "#FBBF24", "#F59E0B", "#78350F", 1.08, 0.10)
        if state == "listening":
            return ("Listening", "#7DD3FC", "#38BDF8", "#0C4A6E", 1.12, 0.14)
        if state == "standby":
            return ("Wake-word standby", "#93C5FD", "#3B82F6", "#172554", 1.03, 0.06)
        return ("Idle", "#A5B4FC", "#818CF8", "#1E1B4B", 1.00, 0.03)

    def _draw_voice_orb(self, state):
        try:
            canvas_width = max(240, int(self.voice_canvas.winfo_width()))
            canvas_height = max(120, int(self.voice_canvas.winfo_height()))
            center_x = canvas_width / 2
            center_y = canvas_height / 2

            label, ring_color, core_color, bg_color, base_scale, pulse = self._state_to_palette(state)
            voice_level = self._get_voice_level()
            pulse_factor = base_scale + (math.sin(self.voice_anim_phase) * pulse) + (voice_level * 0.45)

            core_radius = max(16, 24 * pulse_factor)
            ring_radius = max(core_radius + 16, 42 * pulse_factor)
            ring_width = max(2.0, 2.5 + (voice_level * 3.0))

            self.voice_canvas.coords(self.voice_backdrop, 0, 0, canvas_width, canvas_height)
            self.voice_canvas.itemconfigure(self.voice_backdrop, fill=bg_color)
            self.voice_canvas.coords(
                self.voice_ring,
                center_x - ring_radius,
                center_y - ring_radius,
                center_x + ring_radius,
                center_y + ring_radius,
            )
            self.voice_canvas.coords(
                self.voice_core,
                center_x - core_radius,
                center_y - core_radius,
                center_x + core_radius,
                center_y + core_radius,
            )
            self.voice_canvas.itemconfigure(self.voice_ring, outline=ring_color, width=ring_width)
            self.voice_canvas.itemconfigure(self.voice_core, fill=core_color)
            self.voice_state_var.set(label)
        except tk.TclError:
            return

    def _animate_voice_ui(self):
        if not self.winfo_exists():
            return
        self.voice_anim_phase += 0.30
        state = self._get_visual_state()
        self._decay_voice_level(0.02 if state == "speaking" else 0.04)
        self.voice_visual_state = state
        self._draw_voice_orb(state)
        self.after(60, self._animate_voice_ui)

    def _on_voice_canvas_resize(self, _event):
        self._draw_voice_orb(self.voice_visual_state)

    def _set_status(self, text):
        self.status_var.set(text)

    def _set_busy(self, busy):
        self.busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        self.send_btn.configure(state=state)
        self.voice_btn.configure(state=state)
        self.input_entry.configure(state=state)
        if busy:
            self._set_status("Thinking...")
        elif SLAI.VOICE_INPUT_ENABLED and self.runtime.app_args.handsfree:
            self._set_status("Listening for wake word...")
        else:
            self._set_status("Ready")

    def _run_turn_async(self, user_text):
        def worker():
            try:
                reply = self.runtime.process_message(user_text)
                self.result_queue.put(("reply", reply))
            except Exception as exc:
                self.result_queue.put(("error", f"{exc}"))

        threading.Thread(target=worker, name="slai-gui-turn", daemon=True).start()

    def _submit_user_input(self, user_text, speaker_label="You"):
        if self.busy:
            self.pending_inputs.append((speaker_label, user_text))
            return

        text = user_text.strip()
        if not text:
            return

        self._append_chat(speaker_label, text)
        self._set_busy(True)
        self._run_turn_async(text)

    def _send_message(self):
        user_text = self.input_var.get().strip()
        if not user_text:
            return

        self.input_var.set("")
        self._submit_user_input(user_text, speaker_label="You")

    def _listen_once_with_level(self):
        transcript = None
        error = None
        level = 0.0
        self._mark_listening(True)
        try:
            listener = getattr(SLAI.VOICE_INPUT_ENGINE, "listen_once_with_level", None)
            if callable(listener):
                transcript, error, level = listener(
                    timeout=self.runtime.app_args.stt_timeout,
                    phrase_time_limit=self.runtime.app_args.stt_phrase_time_limit,
                )
            else:
                transcript, error = SLAI.VOICE_INPUT_ENGINE.listen_once(
                    timeout=self.runtime.app_args.stt_timeout,
                    phrase_time_limit=self.runtime.app_args.stt_phrase_time_limit,
                )
        except Exception as exc:
            error = f"Voice input error: {exc}"
        finally:
            self._mark_listening(False)

        self._set_voice_level(level)
        return transcript, error

    def _capture_voice_command_once(self):
        transcript, error = self._listen_once_with_level()
        if error or not transcript:
            return None

        transcript = transcript.strip()
        if not transcript:
            return None

        if not SLAI.is_wake_word_enabled():
            return transcript

        wake_remainder = SLAI.extract_after_any_wake_word(transcript, SLAI.WAKE_WORD_ALIASES)
        if wake_remainder is None:
            return None

        if wake_remainder:
            return wake_remainder.strip()

        follow_up, follow_up_error = self._listen_once_with_level()
        if follow_up_error or not follow_up:
            return None
        return follow_up.strip()

    def _mic_message(self):
        if self.busy:
            return
        if not SLAI.VOICE_INPUT_ENABLED or SLAI.VOICE_INPUT_ENGINE is None:
            self._append_chat("SLAI", "Voice input is disabled.")
            return

        def worker():
            try:
                transcript = self._capture_voice_command_once()
                if not transcript:
                    self.result_queue.put(("error", "No voice command captured."))
                    return
                self.result_queue.put(("voice_input", transcript))
            except Exception as exc:
                self.result_queue.put(("error", f"{exc}"))

        self._append_chat("SLAI", "Listening...")
        threading.Thread(target=worker, name="slai-gui-mic", daemon=True).start()

    def _on_send_enter(self, _event):
        self._send_message()

    def _start_voice_listener(self):
        if not self.runtime.app_args.handsfree:
            return
        if not SLAI.VOICE_INPUT_ENABLED or SLAI.VOICE_INPUT_ENGINE is None:
            return
        if self.voice_listener_thread is not None and self.voice_listener_thread.is_alive():
            return

        self.voice_listener_stop.clear()

        def worker():
            while not self.voice_listener_stop.is_set():
                if self.busy:
                    time.sleep(0.2)
                    continue

                command_text = self._capture_voice_command_once()
                if command_text:
                    self.result_queue.put(("voice_input", command_text))
                    time.sleep(0.2)
                else:
                    time.sleep(0.1)

        self.voice_listener_thread = threading.Thread(target=worker, name="slai-gui-handsfree", daemon=True)
        self.voice_listener_thread.start()

    def _stop_voice_listener(self):
        self.voice_listener_stop.set()
        if self.voice_listener_thread is not None and self.voice_listener_thread.is_alive():
            self.voice_listener_thread.join(timeout=2)
        self.voice_listener_thread = None

    def _poll_results(self):
        try:
            while True:
                kind, payload = self.result_queue.get_nowait()
                if kind == "voice_input":
                    self._submit_user_input(payload, speaker_label="You (voice)")
                elif kind == "reply":
                    self._append_chat("SLAI", payload, speak=True)
                    self._set_busy(False)
                elif kind == "error":
                    self._append_chat("SLAI", f"Error: {payload}")
                    self._set_busy(False)
        except queue.Empty:
            pass

        if not self.busy and self.pending_inputs:
            speaker_label, text = self.pending_inputs.pop(0)
            self._submit_user_input(text, speaker_label=speaker_label)

        self.after(150, self._poll_results)

    def _poll_due_reminders(self):
        try:
            for reminder_text in self.runtime.poll_due_reminders():
                self._append_chat("SLAI", reminder_text, speak=True)
        except Exception:
            pass
        self.after(1000, self._poll_due_reminders)

    def _on_close(self):
        try:
            self._stop_voice_listener()
            self.runtime.shutdown()
        finally:
            self.destroy()


def parse_args():
    parser = argparse.ArgumentParser(description="SLAI Windows Desktop App")
    parser.add_argument("--backend", choices=["nn"], default="nn")
    parser.add_argument("--nn-model-dir", default="artifacts/slai_nn")
    parser.add_argument("--nn-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=220)
    parser.add_argument("--low-resource-mode", action="store_true")
    parser.add_argument("--voice-input", dest="voice_input", action="store_true", default=True)
    parser.add_argument("--no-voice-input", dest="voice_input", action="store_false")
    parser.add_argument("--voice-output", dest="voice_output", action="store_true", default=True)
    parser.add_argument("--no-voice-output", dest="voice_output", action="store_false")
    parser.add_argument("--stt-engine", choices=["google", "whisper"], default="whisper")
    parser.add_argument("--whisper-model", default="tiny")
    parser.add_argument("--whisper-language", default="")
    parser.add_argument("--whisper-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--wake-word", default="alice")
    parser.add_argument("--disable-wake-word", action="store_true")
    parser.add_argument("--stt-timeout", type=int, default=5)
    parser.add_argument("--stt-phrase-time-limit", type=int, default=20)
    parser.add_argument("--autonomous-interval", type=int, default=20)
    parser.add_argument("--enable-autonomy", action="store_true")
    parser.add_argument("--handsfree", dest="handsfree", action="store_true", default=True)
    parser.add_argument("--no-handsfree", dest="handsfree", action="store_false")
    args = parser.parse_args()

    args.disable_autonomy = not args.enable_autonomy
    return args


def main():
    app_args = parse_args()
    runtime = SLAIRuntime(app_args)
    app = SLAIWindowsApp(runtime)
    app.mainloop()


if __name__ == "__main__":
    main()
