"""
Explicit state machine for the session. Without this, barge-in logic turns
into scattered boolean flags that are very easy to get into an inconsistent
state (e.g. "we think we're speaking AND listening at once"). Every state
transition goes through .set() and gets printed, so the terminal output
doubles as a live trace of what the system thinks it's doing — genuinely
useful for debugging why an interrupt did or didn't fire correctly.
"""

from enum import Enum
import threading
from orchestrator.session.websocket_broadcaster import broadcaster


class SessionState(Enum):
    LISTENING = "listening"    # waiting for / accumulating user speech
    THINKING = "thinking"      # STT + agent running, no audio playing yet
    SPEAKING = "speaking"      # TTS audio is playing


class SessionStateMachine:
    def __init__(self, on_state_change=None):
        self._state = SessionState.LISTENING
        self._lock = threading.Lock()
        self.on_state_change = on_state_change

    @property
    def state(self) -> SessionState:
        with self._lock:
            return self._state

    def set(self, new_state: SessionState):
        changed = False
        with self._lock:
            if new_state != self._state:
                print(f"[state] {self._state.value} -> {new_state.value}")
                changed = True
            self._state = new_state
        if changed:
            if self.on_state_change:
                try:
                    self.on_state_change(new_state)
                except Exception:
                    pass
            broadcaster.broadcast({"type": "state", "value": new_state.value})
