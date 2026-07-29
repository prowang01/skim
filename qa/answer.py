"""Chat over a video's content. Palier 3: retrieval, not dump -- each
question retrieves the most relevant transcript segments + frame
descriptions, temporally aligned so the model can reason about actions
(what's said + what's shown together), not just describe frames in
isolation."""

import os
from openai import OpenAI

from index.build_index import IndexItem

CHAT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You answer questions about a video using a *retrieved* excerpt \
of its timestamped content below -- not the full video. The most relevant \
transcript segments and visual frame descriptions for this specific question have \
been selected and temporally aligned:

- (audio) lines: transcribed speech.
- (visual) lines: descriptions of still frames sampled from the video.

Rules:
- When an (audio) line and a (visual) line are close together in time, treat them \
as describing the same moment: reason about what the person is DOING by combining \
what was said with what's shown -- don't just describe the frame in isolation.
- Cite timestamps in [mm:ss] format when your answer relies on a specific moment.
- Because this is a retrieved excerpt, not the full video, the answer might \
genuinely not be present here even if it exists elsewhere in the video -- say so \
rather than guessing or assuming this excerpt is complete.
- Because visual context comes from sampled stills, not continuous motion, be \
honest when a question depends on motion/speed/continuous action that stills \
can't reliably show.

Retrieved context:
{context}
"""


def _format_timestamp(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"


def format_context(items: list[IndexItem]) -> str:
    if not items:
        return "(nothing retrieved)"
    return "\n".join(f"[{_format_timestamp(i.timestamp)}] ({i.kind}) {i.text}" for i in items)


def answer_question(
    items: list[IndexItem],
    question: str,
    history: list[dict] | None = None,
) -> str:
    """Answer a question given the already-retrieved, temporally-aligned
    context items (see index/retrieve.py)."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    context = format_context(items)
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(model=CHAT_MODEL, messages=messages)
    return response.choices[0].message.content
