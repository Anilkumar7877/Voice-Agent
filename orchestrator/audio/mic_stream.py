"""
Continuous microphone capture — unlike silence_detector.py (which records a
single bounded utterance), this runs continuously for the entire session,
producing a steady stream of fixed-size frames regardless of whether the
agent is currently listening or speaking. That's what makes barge-in
possible: the mic never turns off.

Uses sounddevice's callback mode, where the audio backend calls our
function on its own thread every time a new block is ready — we just grab
the samples and drop them in a thread-safe queue for consumption elsewhere.
"""

import queue
import sounddevice as sd
import numpy as np

from orchestrator.audio.vad import FRAME_SAMPLES


class MicStream:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.frame_queue: queue.Queue = queue.Queue()
        self._stream = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            # Overflow/underflow warnings land here — not fatal, but worth
            # printing if you're debugging choppy audio capture.
            print(f"[mic_stream] status: {status}")
        self.frame_queue.put(indata[:, 0].copy())

    def start(self):
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=FRAME_SAMPLES,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get_frame(self, timeout=None) -> np.ndarray:
        """Blocks until a frame is available. timeout in seconds, or None to wait forever."""
        return self.frame_queue.get(timeout=timeout)
