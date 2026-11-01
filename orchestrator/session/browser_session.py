import asyncio
import json
import os
import queue
import tempfile
import threading
import numpy as np
from scipy.io.wavfile import write as wav_write

from orchestrator.config import config
from orchestrator.audio.stt_client import STTClient
from orchestrator.audio.tts_client import TTSClient
from orchestrator.audio.text_chunker import chunk_stream
from orchestrator.audio.vad import VADProcessor, FRAME_SAMPLES
from orchestrator.agent.graph import route_and_retrieve, stream_synthesis
from orchestrator.session.state_machine import SessionStateMachine, SessionState
from orchestrator.interrupt.controller import InterruptController

END_OF_TURN_SILENCE_FRAMES = 20  # ~0.64s of silence ends the user's turn
MAX_SESSION_SECONDS = 300


class BrowserSession:
    def __init__(self, ws, loop):
        self.ws = ws
        self.loop = loop
        self.frames = queue.Queue()
        self._buf = np.zeros(0, dtype=np.int16)
        self.chat_history = []
        self.stt = STTClient()
        self.tts = TTSClient()
        # threshold=0.35 makes VAD more responsive to laptop / mobile browser microphones
        self.vad = VADProcessor(sample_rate=config.SAMPLE_RATE, threshold=0.35)
        self.state = SessionStateMachine(
            on_state_change=lambda new_state: self.emit({"type": "state", "value": new_state.value})
        )
        self.interrupt = InterruptController()
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._vad_loop, daemon=True)

    # ---- plumbing ----
    def start(self):
        self.emit({"type": "state", "value": self.state.state.value})
        self.thread.start()
        self.loop.call_later(MAX_SESSION_SECONDS, lambda: asyncio.ensure_future(self.ws.close()))

    def stop(self):
        self.stopped.set()
        self.interrupt.trigger()

    def emit(self, event):
        asyncio.run_coroutine_threadsafe(self.ws.send(json.dumps(event)), self.loop)

    def send_audio(self, pcm: bytes):
        asyncio.run_coroutine_threadsafe(self.ws.send(pcm), self.loop)

    def handle_message(self, msg):
        if isinstance(msg, bytes):
            self._buf = np.concatenate([self._buf, np.frombuffer(msg, dtype=np.int16)])
            while len(self._buf) >= FRAME_SAMPLES:
                self.frames.put(self._buf[:FRAME_SAMPLES].copy())
                self._buf = self._buf[FRAME_SAMPLES:]
        else:
            try:
                data = json.loads(msg)
                if data.get("type") == "clear_history":
                    self.chat_history.clear()
            except Exception:
                pass

    # ---- same logic as main() in run_barge_in_agent.py ----
    def _vad_loop(self):
        speech_buffer = []
        pre_buffer = []  # rolling buffer of last 5 frames (~160ms) to preserve onset
        silence = 0
        accumulating = False
        in_speech = False

        while not self.stopped.is_set():
            try:
                frame = self.frames.get(timeout=1.0)
            except queue.Empty:
                continue

            # Keep a rolling buffer of 5 frames before speech begins
            if not accumulating:
                pre_buffer.append(frame)
                if len(pre_buffer) > 5:
                    pre_buffer.pop(0)

            ev = self.vad.process_frame(frame)
            if ev and "start" in ev:
                self.emit({"type": "stop_audio"})  # flush audio queued in the browser

            if self.state.state == SessionState.SPEAKING:
                if ev and "start" in ev:
                    self.interrupt.trigger()
                    self.state.set(SessionState.LISTENING)
                    speech_buffer = list(pre_buffer) + [frame]
                    accumulating = True
                    in_speech = True
                    silence = 0
                continue

            if self.state.state == SessionState.THINKING:
                # Do not immediately self-interrupt while thinking on trailing room noise
                continue

            if ev and "start" in ev:
                if not accumulating:
                    speech_buffer = list(pre_buffer) + [frame]
                else:
                    speech_buffer.append(frame)
                accumulating = True
                in_speech = True
                silence = 0
            elif accumulating:
                speech_buffer.append(frame)
                if ev and "end" in ev:
                    in_speech = False
                    silence = 1
                elif in_speech:
                    silence = 0
                else:
                    silence += 1

                if not in_speech and silence >= END_OF_TURN_SILENCE_FRAMES:
                    audio = np.concatenate(speech_buffer, axis=0)
                    speech_buffer = []
                    pre_buffer = []
                    accumulating = False
                    in_speech = False
                    silence = 0
                    self.vad.reset()

                    # Require at least 0.3s (4800 samples) of speech to ignore brief room clicks/pops
                    if len(audio) < 4800:
                        continue

                    self.interrupt.clear()
                    self.state.set(SessionState.THINKING)
                    threading.Thread(target=self._run_turn, args=(audio,), daemon=True).start()

    def _run_turn(self, user_audio):
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav_write(tmp.name, config.SAMPLE_RATE, user_audio)
                path = tmp.name

            try:
                user_text = self.stt.transcribe(path)
            finally:
                try:
                    os.remove(path)
                except Exception:
                    pass

            print(f"[Turn] User text: {user_text}")
            self.emit({"type": "user_transcript", "text": user_text})

            if not user_text or not user_text.strip():
                print("[Turn] No speech recognized, returning to LISTENING")
                self.emit({"type": "info", "message": "No clear speech detected. Please speak clearly into your mic."})
                self.state.set(SessionState.LISTENING)
                return

            if self.interrupt.is_set():
                print("[Turn] Interrupted before routing, returning to LISTENING")
                self.state.set(SessionState.LISTENING)
                return

            result = route_and_retrieve(user_text, self.chat_history)
            print(f"[Turn] Tool: {result.get('tool_choice')}")
            self.emit({"type": "tool_used", "tool": result["tool_choice"], "sources": result["sources"]})

            if self.interrupt.is_set():
                print("[Turn] Interrupted before synthesis, returning to LISTENING")
                self.state.set(SessionState.LISTENING)
                return

            spoken = []

            def tokens():
                for t in stream_synthesis(
                    user_text,
                    result["tool_output"],
                    chat_history=self.chat_history,
                    interrupt_controller=self.interrupt,
                ):
                    self.emit({"type": "agent_token", "text": t})
                    spoken.append(t)
                    yield t

            # Drain any stale queued frames before entering SPEAKING to prevent instant false self-interrupt
            while not self.frames.empty():
                try:
                    self.frames.get_nowait()
                except queue.Empty:
                    break

            # Change state to SPEAKING once tokens start streaming into TTS
            self.state.set(SessionState.SPEAKING)
            completed = self.tts.stream_and_play(
                chunk_stream(tokens()),
                interrupt_controller=self.interrupt,
                on_audio=self.send_audio,
            )
            self.emit({"type": "agent_done"})

            answer = "".join(spoken).strip()
            print(f"[Turn] Answer finished (completed={completed}): {answer[:60]}...")
            if user_text.strip() and answer:
                self.chat_history += [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": answer},
                ]
        except Exception as e:
            print(f"[Turn Error] {e}", flush=True)
            import traceback
            traceback.print_exc()
            self.emit({"type": "error", "message": f"Server Error: {str(e)}"})
        finally:
            self.state.set(SessionState.LISTENING)
