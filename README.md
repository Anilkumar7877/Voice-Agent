# Real-Time Voice Agent with Tool-Calling and Barge-In

A production-style voice assistant that listens continuously, decides
per-turn whether to answer directly, pull from an internal knowledge base,
or search the web — and can be interrupted mid-sentence, the same way a
real conversation works. Built with an accompanying evaluation harness that
tracks tool-routing accuracy, answer quality, and latency across code
changes, tagged by git commit.

This isn't a chatbot with a voice bolted on. It's a real-time system with
the three properties that actually separate a voice agent from a text
agent: streaming (low perceived latency), tool use (grounded, not just
generated, answers), and interruptibility (barge-in).

## What it does

- **Listens continuously** via a real neural VAD (Silero), not a fixed
  recording window — you can start talking whenever you want.
- **Decides what it needs** per question: internal knowledge base (RAG),
  live web search, or a direct answer — routed by an LLM, not hardcoded
  keywords.
- **Speaks while it's still thinking**: the LLM's answer streams
  token-by-token straight into text-to-speech, so playback starts before
  generation finishes.
- **Can be interrupted mid-sentence.** Talk over it, and it stops within a
  fraction of a second and starts listening to you — cooperative
  cancellation across synchronous STT/LLM/TTS APIs, not a hack.
- **Is measured, not just demoed.** An eval harness scores every code
  change against a fixed test set (tool accuracy, LLM-judged answer
  quality, latency) and stores results tagged by git commit, so
  improvements are provable, not just felt.

## Architecture

```
   Continuous mic input
          │
          ▼
  ┌───────────────┐        while SPEAKING, any detected
  │  Silero VAD    │◀──────  speech triggers immediate
  │  (real-time)   │         interrupt + state reset
  └───────┬───────┘
          │ end-of-turn detected
          ▼
  ┌───────────────┐
  │ Groq Whisper   │  (STT)
  └───────┬───────┘
          ▼
  ┌───────────────────────────┐
  │  route_and_retrieve()      │  LLM decides: rag | web_search | direct
  │  ├─ RAG (ChromaDB)         │
  │  ├─ Web search (Tavily)    │
  │  └─ direct (no tool)       │
  └───────────┬───────────────┘
              ▼
  ┌───────────────────────────┐
  │  stream_synthesis()        │  streams the answer token-by-token,
  │  (Llama 3.3 70B via Groq)  │  checking the interrupt flag after
  └───────────┬───────────────┘  every token
              ▼
  ┌───────────────────────────┐
  │  Cartesia Sonic (TTS)      │  streams audio as it's generated,
  │  streaming playback        │  checking the interrupt flag after
  └───────────────────────────┘  every audio chunk
```

Interruption is handled via **cooperative cancellation**: a shared
`threading.Event` checked after every token and every audio chunk, not
`asyncio.Task.cancel()` — the underlying Groq/Cartesia SDKs are
synchronous, so there's no `await` point for asyncio to interrupt. This is
the correct pattern for cancelling blocking work across a sync SDK
boundary, not a workaround.

## Tech stack

| Component | Choice | Why |
|---|---|---|
| STT | Groq (Whisper large-v3-turbo) | Fast, already in the Groq ecosystem |
| LLM | Groq (Llama 3.3 70B) | Fast inference, native streaming |
| TTS | Cartesia Sonic | Built specifically for low time-to-first-audio |
| Agent orchestration | LangGraph | Explicit state machine for routing, not ad-hoc if/else |
| VAD | Silero VAD | Real neural speech detection, not amplitude thresholding |
| Vector store | ChromaDB + sentence-transformers | Standard local RAG stack |
| Web search | Tavily | Purpose-built for agent tool use |
| Eval storage | SQLite | Simple, no infra, tagged by git commit |

## Results (measured, not estimated)

Running the eval harness (`eval/runner.py`) across iterations of this
project produced:

- **Tool-routing accuracy: 100%** (12/12 test cases) after fixing a router
  prompt ambiguity between settled general knowledge and time-sensitive
  facts.
- **Average latency: 3.18s → 1.33s (~58% reduction)**, from fixing a real
  bug where the RAG tool was reloading its embedding model on every single
  call instead of once.
- **Barge-in response time**: sub-second, interrupt-to-silence, confirmed
  via live testing (formal measurement via the eval harness is a planned
  extension — see Roadmap).

See `eval/show_history.py` for the full commit-by-commit comparison.

## Quickstart

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # fill in GROQ_API_KEY, CARTESIA_API_KEY, TAVILY_API_KEY
python scripts/seed_dummy_kb.py
python scripts/run_barge_in_agent.py
```

Speak normally, wait for a response, then talk over it mid-sentence to see
barge-in in action.

## Running the eval harness

```bash
python eval/runner.py          # scores the agent against eval/test_cases/scenarios.json
python eval/show_history.py    # compares results across every commit you've evaluated
```

## Project structure

```
voice-agent/
├── orchestrator/
│   ├── config.py                  # all model/provider config, single source of truth
│   ├── audio/                     # STT, TTS, VAD, mic capture, text chunking
│   ├── agent/
│   │   ├── graph.py                # LangGraph routing + streaming synthesis
│   │   └── tools/                  # RAG tool, web search tool
│   ├── session/                    # state machine (listening/thinking/speaking)
│   └── interrupt/                  # cooperative cancellation controller
├── eval/
│   ├── test_cases/scenarios.json   # test questions + ground truth
│   ├── scorers/                    # tool accuracy, LLM-as-judge
│   ├── runner.py                   # runs eval, saves to SQLite
│   └── show_history.py             # commit-over-commit comparison
└── scripts/                        # entrypoints for each dev phase
```

## Roadmap / known limitations

- **WER and barge-in response time aren't in the eval harness yet** — they
  need prerecorded test audio and simulated interruption, rather than the
  live-mic testing used so far.
- **LLM-as-judge shares the same knowledge-cutoff blind spot as the agent**
  for time-sensitive facts — the eval harness explicitly excludes these
  cases from content grading rather than producing misleading scores (see
  DEVLOG for the specific incident that surfaced this).
- **No acoustic echo cancellation** — barge-in is tested with headphones;
  using speakers without AEC will cause the agent to (correctly) detect
  its own voice as user speech. A WebRTC AEC3-based fix is scoped but not
  yet implemented.

## Development history

The full phase-by-phase build log — every design decision, every bug
found and fixed, in the order it happened — is in [DEVLOG.md](DEVLOG.md).
Worth reading if you want to see the actual debugging process rather than
just the finished result.