# Voice Agent — Phase 1 (Basic Plumbing)

## What this phase does
Single-turn voice pipeline: record audio → transcribe (Groq Whisper) → get an
LLM answer (Groq Llama 3.3 70B) → synthesize speech (Cartesia Sonic) → play it back.

No streaming yet. No barge-in yet. No tools yet. Just prove the round trip works
end to end before we add real-time complexity in Phase 2+.

## Setup

1. Create a virtual environment:
   ```bash
   cd voice-agent
   python3 -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and fill in your keys:
   ```bash
   cp .env.example .env
   ```
   You need:
   - `GROQ_API_KEY` — reuse the one from your RAG project
   - `CARTESIA_API_KEY` — from https://play.cartesia.ai

4. Run the Phase 1 test (text-in, audio-out, no mic yet — proves TTS + LLM work):
   ```bash
   python scripts/test_llm_tts.py
   ```
   This should print the LLM's answer and save `output.wav` — play it to confirm
   Cartesia is working.

5. Run the full loop test (mic-in → STT → LLM → TTS → speaker-out):
   ```bash
   python scripts/test_full_loop.py
   ```
   This records ~5 seconds from your mic, transcribes it, gets an answer, and
   speaks it back. This is your Phase 1 "it works" milestone.

## Project structure so far

```
voice-agent/
├── orchestrator/
│   ├── config.py              # loads API keys, model names — single source of truth
│   ├── audio/
│   │   ├── stt_client.py       # Groq Whisper wrapper (batch, not streaming yet)
│   │   └── tts_client.py       # Cartesia Sonic wrapper (batch, not streaming yet)
│   └── agent/
│       └── llm_client.py       # simple Groq Llama 3.3 call, no tools yet
├── scripts/
│   ├── test_llm_tts.py         # text -> LLM -> TTS, no mic needed
│   └── test_full_loop.py       # mic -> STT -> LLM -> TTS -> speaker
├── requirements.txt
└── .env.example
```

## Why no streaming yet
Streaming (partial transcripts, token-by-token LLM output, streaming TTS
playback) is Phase 2. Building it now, before you've confirmed each individual
piece works, makes debugging much harder — if something breaks you won't know
if it's the streaming logic or the underlying API call. Get the boring
non-streaming version working first.

## Phase 2 — Streaming (added)

Adds:
- `orchestrator/audio/silence_detector.py` — auto-stops recording when you
  pause, instead of a fixed 5-second window. **Not real VAD** — it's naive
  amplitude thresholding. Phase 4 replaces this with Silero VAD for real
  barge-in support.
- `orchestrator/agent/llm_client.py` — added `stream_response()`, which
  yields the LLM's reply token-by-token instead of waiting for the full
  generation.
- `orchestrator/audio/text_chunker.py` — buffers those tokens into
  sentence-sized chunks (so TTS isn't called word-by-word, which sounds
  choppy and wastes API calls).
- `orchestrator/audio/tts_client.py` — added `stream_and_play()`, using
  Cartesia's websocket streaming API, which plays audio as it's generated
  instead of waiting for the full clip.

**Important honesty note:** Groq's Whisper endpoint is batch/REST, not a
streaming websocket — so STT here is still "record a full utterance, then
transcribe it," not true live partial transcription. The latency win in
Phase 2 comes entirely from pipelining LLM generation with TTS playback, not
from streaming STT. If your eval harness later shows STT is your bottleneck,
switching to Deepgram (which does support real streaming STT) is the fix —
that's a config.py change plus a new stt_client, not a rewrite.

Run the Phase 2 test:
```bash
python scripts/test_streaming_loop.py
```
This prints basic latency numbers (time to transcript, time to first LLM
token) — these are the same categories your eval harness will formalize in
Phase 5, so pay attention to what "feels slow" here.

**A note on the Cartesia websocket code**: streaming TTS SDKs change their
method signatures between versions more often than REST APIs do. If
`stream_and_play()` throws an `AttributeError` or unexpected `TypeError`,
check `pip show cartesia` and cross-reference the version against
https://docs.cartesia.ai — paste me the exact error and installed version
and I'll fix the call to match.

## Next phase
Phase 3: tool-calling. The agent will decide per-turn whether to answer
directly, query your existing RAG/ChromaDB store, or run a web search —
using LangGraph, reusing the same routing pattern from your Research
Assistant project.
