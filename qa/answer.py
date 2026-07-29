"""Chat over a video's fused audio transcript + visual frame descriptions.
Palier 2: still dumps the full fused context into the LLM -- no retrieval
yet. That comes in a later palier."""

import os
from openai import OpenAI

from ingest.transcribe import Segment
from ingest.describe import FrameDescription

CHAT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You answer questions about a video using the timestamped context \
below, which fuses two sources:

- (audio) lines: transcribed speech.
- (visual) lines: descriptions of still frames sampled from the video (not \
continuous video -- treat them as snapshots at that instant).

Rules:
- Cite timestamps in [mm:ss] format when your answer relies on a specific moment.
- If a question needs both what was said and what was shown, fuse both sources \
in your answer rather than picking one.
- If the context doesn't contain the answer, say so plainly instead of guessing.
- Because visual context comes from sampled stills, not continuous motion, be \
honest when a question depends on motion/speed/continuous action that stills \
can't reliably show -- say so rather than guessing from a still.

Context:
{context}
"""


def _format_timestamp(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"


def format_context(
    segments: list[Segment],
    frame_descriptions: list[FrameDescription] | None = None,
) -> str:
    """Fuse transcript segments and frame descriptions into one timestamped,
    source-tagged context, sorted chronologically."""
    lines = [(s.start, "audio", s.text) for s in segments]
    lines += [
        (f.timestamp, "visual", f.description)
        for f in (frame_descriptions or [])
        if f.description
    ]
    lines.sort(key=lambda item: item[0])
    return "\n".join(f"[{_format_timestamp(t)}] ({source}) {text}" for t, source, text in lines)


def answer_question(
    segments: list[Segment],
    question: str,
    frame_descriptions: list[FrameDescription] | None = None,
    history: list[dict] | None = None,
) -> str:
    """Answer a question about the video, given prior chat turns for context."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    context = format_context(segments, frame_descriptions)
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(model=CHAT_MODEL, messages=messages)
    return response.choices[0].message.content
