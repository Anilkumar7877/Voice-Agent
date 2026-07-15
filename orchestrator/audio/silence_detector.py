"""
Naive silence detection for auto-stopping the recording when the user stops
talking, instead of always recording a fixed window.

IMPORTANT — be honest about what this is: this is amplitude/energy-based
silence detection, NOT real Voice Activity Detection (VAD). It can't tell
speech apart from background noise as reliably as a trained VAD model.
Phase 4 replaces this with Silero VAD (a real neural model) when we build
barge-in / interruption handling, which needs to be much more accurate.
This version is good enough to make Phase 2 feel responsive; don't mistake
it for the real thing when explaining the project later.
"""

import numpy as np
import sounddevice as sd


def record_until_silence(
    sample_rate: int = 16000,
    silence_threshold: float = 500.0,
    silence_duration_sec: float = 1.2,
    max_duration_sec: float = 30.0,
    chunk_ms: int = 100,
) -> np.ndarray:
    """
    Record from the mic until `silence_duration_sec` of near-silence is
    detected, or `max_duration_sec` is reached (safety cap).

    silence_threshold: RMS amplitude below which audio is considered silence.
        This is a rough value tuned for typical laptop mics at normal speaking
        volume — you may need to raise/lower it depending on your mic and
        room noise. Print the RMS values if it's cutting you off too early
        or too late, and adjust.
    """
    chunk_samples = int(sample_rate * chunk_ms / 1000)
    silence_chunks_needed = int(silence_duration_sec * 1000 / chunk_ms)
    max_chunks = int(max_duration_sec * 1000 / chunk_ms)

    frames = []
    silent_count = 0
    started_talking = False

    print("Listening... (speak now, it'll stop automatically when you pause)")

    with sd.InputStream(
        samplerate=sample_rate, channels=1, dtype="int16", blocksize=chunk_samples
    ) as stream:
        for _ in range(max_chunks):
            chunk, _ = stream.read(chunk_samples)
            frames.append(chunk.copy())

            rms = np.sqrt(np.mean(chunk.astype(np.float64) ** 2))

            if rms > silence_threshold:
                started_talking = True
                silent_count = 0
            elif started_talking:
                silent_count += 1

            if started_talking and silent_count >= silence_chunks_needed:
                break

    print("Done listening.")
    return np.concatenate(frames, axis=0)
