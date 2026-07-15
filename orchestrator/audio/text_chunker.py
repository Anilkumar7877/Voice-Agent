"""
Buffers a stream of small text fragments (as they come token-by-token from
the LLM) into sentence-ish chunks before handing them to TTS.

Why this matters: sending TTS one word at a time sounds choppy and wastes
API calls; waiting for the entire LLM response defeats the point of
streaming. The middle ground — flush on sentence boundaries (. ! ? and also
commas for long sentences) — is what production voice agents do, because it
keeps the "first audio" latency low while still sounding natural.
"""

SENTENCE_ENDERS = (".", "!", "?")
CLAUSE_ENDERS = (",", ";", ":")
MAX_BUFFER_CHARS = 200  # safety flush so a long unpunctuated stream doesn't stall TTS


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
