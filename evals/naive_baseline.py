"""Naive baseline for eval comparison: dump the full transcript + all frame
descriptions into the prompt, no retrieval -- this is what Palier 1/2 did
before Palier 3 added retrieval. Used only to measure how much retrieval
and the Palier 4 blind-spot polish actually improved things."""

import os
from openai import OpenAI

from index.build_index import Index
from qa.answer import format_context, CHAT_MODEL

NAIVE_SYSTEM_PROMPT = """You answer questions about a video using its full timestamped \
content below:

- (audio) lines: transcribed speech.
- (visual) lines: descriptions of still frames sampled from the video.

Rules:
- Cite timestamps in [mm:ss] format when your answer relies on a specific moment.
- If a question needs both what was said and what was shown, fuse both sources.
- If the content below doesn't contain the answer, say so plainly instead of guessing.
- Because visual context comes from sampled stills, not continuous motion, be \
honest when a question depends on motion/speed/continuous action that stills \
can't reliably show.

Content:
{context}
"""


def answer_naive(index: Index, question: str, history: list[dict] | None = None) -> str:
    """Answer using the entire index (no retrieval) -- the naive baseline."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    context = format_context(index.items)
    messages = [{"role": "system", "content": NAIVE_SYSTEM_PROMPT.format(context=context)}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(model=CHAT_MODEL, messages=messages)
    return response.choices[0].message.content
