"""
Real Voice Activity Detection using Silero VAD — a small pretrained neural
model, not amplitude thresholding. This replaces silence_detector.py's
naive approach for anything that needs to work while OTHER audio (the
agent's own TTS playback) might be happening at the same time, where a
simple volume threshold would be useless (the agent's own voice would
constantly trigger it as "input").

Silero VAD requires audio in fixed-size frames: exactly 512 samples per
call at 16kHz (32ms). Feed it anything else and it'll error — that's why
mic_stream.py captures audio in blocks of exactly this size.

First run downloads/loads the model — if you're behind a restrictive
firewall and load_silero_vad() fails to fetch anything, check
https://github.com/snakers4/silero-vad for alternate load methods
(torch.hub.load with a local path).
"""

import numpy as np
import torch
from silero_vad import load_silero_vad, VADIterator

FRAME_SAMPLES = 512  # required frame size at 16kHz — do not change without also changing mic_stream.py


class VADProcessor:
    def __init__(self, sample_rate: int = 16000, threshold: float = 0.5):
        """
        threshold: speech probability cutoff (0-1). Lower = more sensitive
        (catches quieter speech, but more false positives from noise).
        0.5 is Silero's own recommended default — tune only if you're
        seeing clear false triggers or missed speech in practice.
        """
        self.model = load_silero_vad()
        self.iterator = VADIterator(self.model, sampling_rate=sample_rate, threshold=threshold)
        self.sample_rate = sample_rate

    def process_frame(self, frame_int16: np.ndarray):
        """
        Feed one 512-sample int16 frame. Returns:
        - {'start': <sample_idx>} the frame where speech was judged to begin
        - {'end': <sample_idx>} the frame where speech was judged to end
        - None if no state change (still speaking, or still silent)
        """
        audio_float32 = frame_int16.astype(np.float32) / 32768.0
        tensor = torch.from_numpy(audio_float32)
        return self.iterator(tensor, return_seconds=False)

    def reset(self):
        """Call between logical sessions/turns — the model carries internal
        state between frames for accuracy, and that state needs clearing so
        one turn's tail doesn't bleed into the next turn's detection."""
        self.iterator.reset_states()
