"""
Full barge-in support MERGED with tool-routing agent. Continuous mic
capture + Silero VAD + cooperative interruption, now with the agent
actually calling RAG/web-search tools instead of a plain LLM.

Architecture (deliberately threads, not asyncio — see README for why):
- MAIN THREAD: runs the VAD monitor loop. Continuously pulls frames from
  MicStream, feeds them to VADProcessor, and reacts to speech-start/
  speech-end events depending on the current SessionState.
- WORKER THREAD (spawned per turn): does the actual work for one user
  turn: STT -> route_and_retrieve() (fast, blocking: decide + call the
  right tool) -> stream_synthesis() (streaming, interruptible: generate
  the spoken answer token by token) -> TTS playback.

State transitions:
  LISTENING --(VAD: speech ends)--> spawn turn worker --> THINKING
  THINKING  --(worker starts TTS)--> SPEAKING
  SPEAKING  --(VAD: speech starts)--> interrupt_controller.trigger(),
                                        LISTENING (start accumulating the
                                        NEW utterance immediately)
  SPEAKING  --(worker finishes normally)--> LISTENING

WHY SPLIT ROUTING FROM SYNTHESIS: graph.invoke() (the full LangGraph
pipeline) runs route -> tool -> synthesize as one blocking call and
returns the complete answer at once — no point to check an interrupt flag
mid-generation. So this script uses route_and_retrieve() (just the routing
+ tool call, quick and left as blocking — a turn being "uninterruptible"
for under ~2s while it decides what to do is an acceptable tradeoff) and
then stream_synthesis() (the actual answer generation, which DOES stream
token-by-token and DOES check the interrupt flag) — both from
orchestrator/agent/graph.py, alongside the original VoiceAgent/build_graph
which is still used by test_agent_graph.py and the eval harness where
streaming doesn't matter.

Run: python scripts/run_barge_in_agent.py
Press Ctrl+C to stop.
"""

import sys
import os
import threading
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.io.wavfile import write as wav_write

from orchestrator.config import config
from orchestrator.audio.stt_client import STTClient
from orchestrator.audio.tts_client import TTSClient
from orchestrator.audio.text_chunker import chunk_stream
from orchestrator.audio.vad import VADProcessor
from orchestrator.audio.mic_stream import MicStream
from orchestrator.agent.graph import route_and_retrieve, stream_synthesis
from orchestrator.session.state_machine import SessionStateMachine, SessionState
from orchestrator.interrupt.controller import InterruptController
from orchestrator.session.websocket_broadcaster import broadcaster

# How much silence (in consecutive VAD-silent frames) ends a turn.
# Each frame = 512 samples @ 16kHz = 32ms. So 25 frames = 25 * 32ms = ~0.8s
# of REAL post-speech silence (tracked via the in_speech flag below — VAD
# returns None during active speech too, so None alone never counts as
# silence; only None AFTER a real 'end' event counts).
END_OF_TURN_SILENCE_FRAMES = 25  # ~0.8s


def run_turn(user_audio: np.ndarray, stt, tts, state, interrupt_controller, chat_history):
    """
    Runs in a background thread: STT -> route_and_retrieve (tool routing,
    quick, not interruptible mid-call) -> stream_synthesis (streaming,
    interruptible token-by-token) -> streaming TTS playback.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_write(tmp.name, config.SAMPLE_RATE, user_audio)
        input_path = tmp.name

    user_text = stt.transcribe(input_path)
    print(f"\nYou said: {user_text}")
    broadcaster.broadcast({"type": "user_transcript", "text": user_text})

    if not user_text or interrupt_controller.is_set():
        state.set(SessionState.LISTENING)
        return

    result = route_and_retrieve(user_text, chat_history)
    print(f"[Tool used: {result['tool_choice']}]", end="")
    if result["sources"]:
        print(f"  Sources: {result['sources']}", end="")
    print()
    broadcaster.broadcast({"type": "tool_used", "tool": result["tool_choice"], "sources": result["sources"]})

    # The routing+retrieval call above was blocking — check whether the
    # user started talking again WHILE it was running, before we commit to
    # speaking a (possibly now-irrelevant) answer.
    if interrupt_controller.is_set():
        state.set(SessionState.LISTENING)
        return

    print("Agent: ", end="", flush=True)

    spoken_tokens = []

    def printing_token_stream():
        for token in stream_synthesis(
            user_text, result["tool_output"], chat_history=chat_history, interrupt_controller=interrupt_controller
        ):
            print(token, end="", flush=True)
            broadcaster.broadcast({"type": "agent_token", "text": token})
            spoken_tokens.append(token)
            yield token

    text_chunks = chunk_stream(printing_token_stream())

    state.set(SessionState.SPEAKING)
    completed = tts.stream_and_play(text_chunks, interrupt_controller=interrupt_controller)
    print()
    broadcaster.broadcast({"type": "agent_done"})

    spoken_answer = "".join(spoken_tokens).strip()
    if user_text.strip() and spoken_answer:
        chat_history.append({"role": "user", "content": user_text})
        chat_history.append({"role": "assistant", "content": spoken_answer})

    if completed:
        state.set(SessionState.LISTENING)
    # If NOT completed, the VAD monitor loop already moved state to
    # LISTENING itself when it detected the interruption — don't overwrite
    # it back here.


def main():
    config.validate()

    # Initialize chat history memory list
    chat_history = []
    broadcaster.on_clear_history = lambda: chat_history.clear()

    # Start WebSocket broadcaster
    broadcaster.start()

    stt = STTClient()
    tts = TTSClient()
    vad = VADProcessor(sample_rate=config.SAMPLE_RATE)
    mic = MicStream(sample_rate=config.SAMPLE_RATE)
    state = SessionStateMachine()
    interrupt_controller = InterruptController()

    mic.start()
    print("Listening continuously. Speak anytime — you can also interrupt the agent mid-response.")
    print("Press Ctrl+C to stop.\n")

    speech_buffer = []
    silence_frame_count = 0
    accumulating = False
    in_speech = False  # tracks whether we're currently INSIDE a detected
                        # speech segment, per VAD's own start/end events —
                        # NOT inferred from None, which just means "no
                        # transition this frame" and says nothing about
                        # whether that frame was speech or silence.

    try:
        while True:
            frame = mic.get_frame(timeout=1.0)
            vad_event = vad.process_frame(frame)

            if state.state == SessionState.SPEAKING:
                if vad_event and "start" in vad_event:
                    interrupt_controller.trigger()
                    state.set(SessionState.LISTENING)
                    speech_buffer = [frame]
                    accumulating = True
                    in_speech = True
                    silence_frame_count = 0
                continue

            if state.state == SessionState.THINKING:
                if vad_event and "start" in vad_event:
                    interrupt_controller.trigger()
                continue

            # state == LISTENING: accumulate speech, detect end-of-turn
            if vad_event and "start" in vad_event:
                if not accumulating:
                    speech_buffer = [frame]
                else:
                    speech_buffer.append(frame)  # resuming after a brief dip — keep the buffer
                accumulating = True
                in_speech = True
                silence_frame_count = 0
            elif accumulating:
                speech_buffer.append(frame)

                if vad_event and "end" in vad_event:
                    in_speech = False
                    silence_frame_count = 1  # start counting from this frame
                elif in_speech:
                    # Still inside an active speech segment per VAD's own
                    # state — None here means "still speaking," not
                    # "silence." Do NOT count this toward end-of-turn.
                    silence_frame_count = 0
                else:
                    # Genuinely past the 'end' event — this None really is
                    # silence, so it's safe to count it.
                    silence_frame_count += 1

                if not in_speech and silence_frame_count >= END_OF_TURN_SILENCE_FRAMES:
                    user_audio = np.concatenate(speech_buffer, axis=0)
                    speech_buffer = []
                    accumulating = False
                    in_speech = False
                    silence_frame_count = 0
                    vad.reset()
                    interrupt_controller.clear()

                    state.set(SessionState.THINKING)
                    turn_thread = threading.Thread(
                        target=run_turn,
                        args=(user_audio, stt, tts, state, interrupt_controller, chat_history),
                        daemon=True,
                    )
                    turn_thread.start()

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        mic.stop()


if __name__ == "__main__":
    main()
