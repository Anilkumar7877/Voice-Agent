"""
Buffers a stream of small text fragments (as they come token-by-token from
the LLM) into sentence-ish chunks before handing them to TTS.

Why this matters: sending TTS one word at a time sounds choppy and wastes
API calls; waiting for the entire LLM response defeats the point of
streaming. The middle ground — flush on sentence boundaries (. ! ? and also
commas for long sentences) — is what production voice agents do, because it
keeps the "first audio" latency low while still sounding natural.
"""

import re

SENTENCE_ENDERS = (".", "!", "?")
CLAUSE_ENDERS = (",", ";", ":")
MAX_BUFFER_CHARS = 200  # safety flush so a long unpunctuated stream doesn't stall TTS


def chunk_text(text: str):
    """
    Split a COMPLETE string (not a live stream) into sentence-sized chunks
    for TTS. Use this for output that's already fully generated (e.g. the
    agent graph's final_answer, which comes back all at once since
    graph.invoke() is blocking) — as opposed to chunk_stream() below, which
    is for genuinely live token-by-token output.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]


def split_into_sentences(text: str) -> list:
    """
    Split an already-complete string into sentence-sized chunks for TTS.

    Different from chunk_stream() below: this is for text you already have
    in full (e.g. the agent graph's final_answer, since graph.invoke() is
    blocking and doesn't give you tokens incrementally). Splitting it into
    sentences still lets TTS start playing sentence 1 while synthesizing
    sentence 2, instead of waiting for the whole answer to be sent as one
    TTS call — smaller latency win than true token streaming, but real.
    """
    sentences = []
    buffer = ""
    for char in text:
        buffer += char
        if buffer.rstrip().endswith(SENTENCE_ENDERS):
            sentences.append(buffer.strip())
            buffer = ""
    if buffer.strip():
        sentences.append(buffer.strip())
    return sentences if sentences else [text]


def chunk_stream(token_stream):
    """
    Takes an iterator of small text fragments (e.g. from LLMClient.stream_response)
    and yields larger, sentence-ish chunks suitable for TTS.
    """
    buffer = ""
    for fragment in token_stream:
        buffer += fragment

        if buffer.rstrip().endswith(SENTENCE_ENDERS):
            yield buffer.strip()
            buffer = ""
        elif len(buffer) >= MAX_BUFFER_CHARS and buffer.rstrip().endswith(CLAUSE_ENDERS):
            yield buffer.strip()
            buffer = ""
        elif len(buffer) >= MAX_BUFFER_CHARS * 2:
            # Safety valve: no punctuation for a very long stretch, flush anyway
            yield buffer.strip()
            buffer = ""

    if buffer.strip():
        yield buffer.strip()
