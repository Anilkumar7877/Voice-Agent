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

## Phase 3 — Agent tool-calling (added)

Adds:
- `orchestrator/agent/tools/rag_tool.py` — RAG tool backed by a **dummy**
  ChromaDB knowledge base (5 placeholder HR/engineering docs). Uses the
  same embedding model (`all-MiniLM-L6-v2`) as a real production RAG
  setup, so swapping in your actual Document Q&A project's store later is
  just changing `CHROMA_PERSIST_DIR` / `CHROMA_COLLECTION_NAME` in
  `config.py` — not a rewrite.
- `orchestrator/agent/tools/web_search_tool.py` — Tavily web search wrapper.
- `orchestrator/agent/graph.py` — the actual agent: a LangGraph state
  machine with a `route` node (LLM decides: rag / web_search / direct),
  the corresponding tool node, and a `synthesize` node that turns the
  result into a short, voice-friendly answer.
- `scripts/seed_dummy_kb.py` — run once to build the dummy Chroma
  collection.
- `scripts/test_agent_graph.py` — **text-only** test of the agent's
  routing + tool-calling, no mic/STT/TTS involved. This is the fast
  iteration loop — get routing right here before wiring into voice.

### Setup for this phase
```bash
pip install -r requirements.txt   # picks up langgraph, chromadb, sentence-transformers, tavily-python
```
Add `TAVILY_API_KEY` to your `.env` (free tier, 1,000 searches/month, no
card — get one at https://app.tavily.com).

Then:
```bash
python scripts/seed_dummy_kb.py      # builds the dummy knowledge base
python scripts/test_agent_graph.py   # runs 4 test questions through the agent
```
You should see each question correctly routed to `rag`, `web_search`, or
`direct`, with sources shown for the tool-based ones. If routing looks
wrong for a question, the fix is almost always in `ROUTER_PROMPT` inside
`graph.py` — tighten the tool descriptions, not the routing logic itself.

### What's NOT done yet in this phase
~~The agent graph is not yet wired into the voice pipeline~~ — now added:
`scripts/test_voice_agent.py` wires `VoiceAgent` into the full mic → STT →
agent → TTS loop.

**Known tradeoff, on purpose:** `VoiceAgent.run()` (via `graph.invoke()`) is
blocking — router, tool call, and synthesis all complete before you get an
answer back, so you lose the "starts speaking before it's done thinking"
benefit that `test_streaming_loop.py` has. What's recovered instead: the
completed answer is split into sentences (`text_chunker.chunk_text`)
and streamed to TTS, so playback starts on sentence 1 while sentence 2 is
still being synthesized — a smaller win than true token streaming, but real.

This is intentionally left as a measurable gap rather than hidden — once
the eval harness (Phase 5) exists, compare `test_streaming_loop.py` vs
`test_voice_agent.py` latency numbers directly. That comparison ("added
tool-calling, here's the measured latency cost, here's what streaming
recovered") is exactly the kind of before/after story that makes a strong
resume line, not just a vague "improved performance" claim.

Run it:
```bash
python scripts/test_voice_agent.py
```

## Next phase
Phase 4: barge-in / interruption handling with real Silero VAD — the
hardest phase. After that, Phase 5 is the eval harness, which will finally
let you measure and report concrete before/after numbers for your resume.

## Phase 4 — Barge-in / Interruption (added)

This is the hardest phase, so a bit more explanation than usual.

### What's new
- `orchestrator/audio/vad.py` — real Silero VAD (a small pretrained neural
  model), replacing `silence_detector.py`'s naive amplitude thresholding
  for anything that needs to work *while other audio is playing* — a
  volume threshold would be useless here since the agent's own TTS
  playback would constantly trigger it.
- `orchestrator/audio/mic_stream.py` — continuous mic capture in fixed
  512-sample frames (Silero's required frame size at 16kHz), running for
  the whole session rather than one bounded recording per turn.
- `orchestrator/session/state_machine.py` — explicit
  `LISTENING / THINKING / SPEAKING` states. Every transition prints to the
  terminal, so you get a live trace of what the system thinks it's doing.
- `orchestrator/interrupt/controller.py` — a cooperative cancellation flag
  (see "why threads, not asyncio" below).
- `scripts/run_barge_in_agent.py` — the full real-time loop tying it
  together.

### Why threads + cooperative cancellation, not asyncio
Early in planning this phase I said "asyncio, not threads" for
cancellation. That was wrong, and worth correcting rather than quietly
fixing: Cartesia's and Groq's Python SDKs are **synchronous** — there's no
`await` point inside their blocking calls for `asyncio.Task.cancel()` to
actually interrupt. You cannot cleanly force-cancel a blocking network call
from outside in Python.

The correct, standard pattern for this situation is **cooperative
cancellation**: `InterruptController` is just a `threading.Event`. The TTS
playback loop checks it after every audio chunk; the LLM streaming
generator checks it after every token. When barge-in fires, we don't
force-kill anything — we just flip a flag, and the loop notices within
tens of milliseconds and exits on its own. This is genuinely how
production systems handle cancellation across a synchronous SDK boundary,
not a compromise.

### Architecture
```
MAIN THREAD (VAD monitor loop)          WORKER THREAD (spawned per turn)
─────────────────────────────           ──────────────────────────────
continuously reads 512-sample
frames from MicStream, runs
Silero VAD on each one
        │
        ▼
state == LISTENING:
  accumulate speech frames,
  detect end-of-turn (silence
  run) → spawn worker thread ────────▶  STT (Groq Whisper)
  → state = THINKING                    │
        │                               ▼
state == THINKING:                     stream_response() token by token,
  still watching for speech-start      checking interrupt_controller
  (barge-in before audio even          after every token
  starts playing)                       │
        │                               ▼ (once first chunk ready)
state == SPEAKING:                     state = SPEAKING
  if VAD detects speech-start:                │
    interrupt_controller.trigger()            ▼
    state = LISTENING                  stream_and_play() writes audio
    start accumulating the             chunk by chunk, checking
    NEW utterance immediately          interrupt_controller after each one
    (starting with the frame                  │
    that triggered it, so we           if interrupted: stop immediately,
    don't lose the first                return False
    syllable)                          if completed: state = LISTENING
```

### Scope decision: built on Phase 2's pipeline, not Phase 3's agent
~~Barge-in is wired into the streaming LLM + TTS pipeline~~ — **now merged
with tool-routing, see below.**

### Setup for this phase
```bash
pip install -r requirements.txt   # adds silero-vad
```
No new API keys needed — Silero VAD's model ships inside the pip package
itself (no network fetch at runtime, confirmed while building this).

```bash
python scripts/run_barge_in_agent.py
```
Speak normally, wait for a response, then **try talking over the agent
while it's mid-sentence** — it should stop within a fraction of a second
and start listening to your new input. Watch the `[state] ... -> ...`
lines in the terminal; they're your best debugging tool if something feels
off.

### What to expect the first time you run this
- The end-of-turn silence detection (`END_OF_TURN_SILENCE_FRAMES = 25`,
  ~0.8s) may need tuning — if it cuts you off mid-sentence or waits too
  long after you're done talking, adjust that constant.
- Silero's `threshold=0.5` (in `vad.py`) may need tuning for your mic/room
  — if it's not detecting your speech, or false-triggering on background
  noise, that's the value to adjust first.
- The very first interruption you try is the real test of this whole
  phase. If it doesn't stop the audio near-instantly, tell me exactly what
  happened (did it finish the sentence? ignore you completely? crash?) —
  that'll point to which piece needs fixing.

## Next phase
Phase 5: the eval harness. This is what turns "I built a voice agent" into
"I built a voice agent and can show measured improvements across N
iterations" — WER, latency (including barge-in response time), tool-call
accuracy, and cost per turn, tracked against git commits. This is the
single highest-leverage remaining piece for your resume framing.

## Phase 5 — Eval Harness, first cut (added)

### Scope of this cut
This version scores **tool-call accuracy**, **task success** (LLM-as-judge),
and **latency**, run against the Phase 3 agent directly via text (no audio
needed — fast to run, good for iterating on prompts/routing). WER and
barge-in response time need actual audio test clips and simulated
interruption — that's a natural follow-up once this core version proves
useful, not part of this cut.

### What's new
- `eval/test_cases/scenarios.json` — 12 test questions (5 rag, 4
  web_search, 3 direct), each with an `expected_tool` and a `ground_truth`
  reference answer for scoring.
- `eval/scorers/tool_accuracy_scorer.py` — rule-based, no LLM: did the
  agent pick the tool we expected.
- `eval/scorers/llm_judge_scorer.py` — LLM-as-judge: does the agent's
  actual answer match the ground truth semantically (scored
  correct/partial/incorrect, not exact string match — wording can differ
  and still be right).
- `eval/results_db.py` — SQLite storage, every row tagged with the current
  git commit hash (via `git rev-parse --short HEAD`). This is what makes
  "compare before and after a change" possible — run the eval, make a
  change, commit, run the eval again, and the two results sit side by side
  tagged by commit.
- `eval/runner.py` — runs every scenario through `VoiceAgent.run()`,
  scores each, prints a live summary, saves everything to SQLite.
- `eval/show_history.py` — prints the aggregate metrics per commit you've
  evaluated, plus a first-vs-most-recent comparison. **This is the table
  your resume line comes from.**

### Running it
```bash
python eval/runner.py          # run the eval against your current code
# ... make a change, commit it ...
python eval/runner.py          # run again — new commit hash tags this run
python eval/show_history.py    # see the before/after comparison
```

### Verified before handing this over
I ran the full DB write → aggregate → comparison flow with simulated data
(two fake "commits," one deliberately worse than the other) and confirmed
`show_history.py` correctly shows the improvement — 60%→100% tool
accuracy, 2.5s→1.8s latency, formatted exactly like your real results will
look. The scorer logic and runner haven't been run against your real Groq
key yet — that's the next step.

### A note on the LLM-judge scorer
It's not perfectly deterministic — you may see slightly different
correct/partial/incorrect calls on two runs of *identical* code, since
that's inherent to using an LLM as a judge. This is worth knowing rather
than treating single eval runs as absolute truth: look at trends across
several runs, not one run in isolation, especially for borderline cases.

## Next phase
Phase 6: polish — resume framing, demo prep, and closing the loop on the
things intentionally deferred earlier (WER scoring with real audio,
barge-in response time measurement, merging tool-routing with the
streaming/barge-in pipeline).

## Phase 5.5 — Merging barge-in with tool-routing (added)

This closes the gap Phase 4 deliberately left open: `run_barge_in_agent.py`
now actually calls the RAG/web-search tools, not a plain LLM.

### The core architectural move
`graph.invoke()` (the full LangGraph pipeline from Phase 3) runs
route → tool → synthesize as ONE blocking call — there's no point inside
it to check an interrupt flag mid-generation. So the agent is split into
two pieces, both in `orchestrator/agent/graph.py`:

- **`route_and_retrieve(question)`** — runs just the routing + tool-call
  part (calls the same `route_node`/`rag_node`/`web_search_node`/
  `direct_node` functions the full graph uses, just without going through
  LangGraph's runtime). This stays blocking — it's usually one or two
  network calls, done in under ~2s, and making *this* part interruptible
  wasn't worth the complexity.
- **`stream_synthesis(question, tool_output, interrupt_controller)`** —
  the actual answer generation, now streaming token-by-token with the
  same cooperative-interrupt check as `LLMClient.stream_response`. This
  is the part that can be cut off mid-sentence during barge-in.

`run_barge_in_agent.py`'s `run_turn()` now calls these two in sequence,
with an interrupt check in between — if the user starts talking again
while `route_and_retrieve` is running (before any audio has even started),
the turn is abandoned before wasting a TTS call on an answer nobody wants.

The original `VoiceAgent` / `build_graph()` (the full blocking pipeline)
is untouched and still used by `test_agent_graph.py` and the eval harness,
where streaming doesn't matter and the simpler interface is preferable.

### Verified before handing over
Tested both new functions with mocked Groq responses (no real API calls):
confirmed `route_and_retrieve` correctly routes and returns tool output,
confirmed `stream_synthesis` correctly yields tokens as they arrive, and
confirmed it correctly **stops mid-generation** when the interrupt
controller fires partway through — the exact behavior barge-in depends on.

### What to test on your end
Run `python scripts/run_barge_in_agent.py` again — same as Phase 4, but
now ask something that should hit the dummy KB or web search (not just
generic small talk), and confirm:
1. It prints `[Tool used: rag]` (or `web_search`/`direct`) before speaking
2. Barge-in still works — interrupting mid-sentence should feel identical
   to Phase 4's test, just now backed by real tool calls instead of a
   plain LLM

## Next phase
Phase 6: polish — resume framing, demo prep, and closing the loop on the
remaining deferred items (WER scoring with real audio, barge-in response
time measurement in the eval harness).
