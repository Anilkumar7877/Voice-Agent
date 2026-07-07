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

## Next phase
Once `test_full_loop.py` works reliably, we move to Phase 2: streaming STT
(partial transcripts as you talk), streaming LLM tokens, and streaming TTS
playback — so the agent starts responding before you've even finished the
full turn.
