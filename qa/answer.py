"""Chat over a video transcript. Palier 1: dump the full transcript into the
LLM's context — no retrieval yet. That comes in a later palier."""

import os
from openai import OpenAI

from ingest.transcribe import Segment

CHAT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You answer questions about a video using only the timestamped \
transcript of its spoken audio provided below. Rules:

- Cite timestamps in [mm:ss] format when your answer relies on a specific moment.
- If the transcript doesn't contain the answer, say so plainly instead of guessing.
- You only have the spoken audio, not the video's visuals — if a question depends \
on something visual (on-screen text, what's shown, gestures), say you can't see the \
video and can only speak to what was said.

Transcript:
{transcript}
"""


def _format_timestamp(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"


def format_transcript(segments: list[Segment]) -> str:
    lines = [f"[{_format_timestamp(s.start)}] {s.text}" for s in segments]
    return "\n".join(lines)


def answer_question(
    segments: list[Segment],
    question: str,
    history: list[dict] | None = None,
) -> str:
    """Answer a question about the video, given prior chat turns for context."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    transcript = format_transcript(segments)
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(transcript=transcript)}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(model=CHAT_MODEL, messages=messages)
    return response.choices[0].message.content
