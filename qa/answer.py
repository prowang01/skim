"""Chat over a video's content: fuses retrieved transcript segments and
frame descriptions, temporally aligned, so the model reasons about actions
(what's said + what's shown together) rather than describing frames in
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
- A line marked "no visual/audio context within Ns" has no nearby coverage from \
the other modality -- treat that moment as a gap, not a blank canvas to extrapolate from.

Know your blind spots -- say so explicitly rather than guessing when:
- A question depends on motion, speed, or continuous action (e.g. "fast or slow?", \
"how smoothly?"). You only ever have transcribed speech and sampled still frames, \
never continuous video, so you cannot reliably judge motion or speed. Say this \
plainly instead of inferring it from wording or a single still.
- A line is flagged with no nearby context from the other modality -- a brief \
action or detail there may simply not have been captured by a sampled frame or \
caught in speech. Flag the uncertainty instead of confidently filling the gap.
- The retrieved excerpt below doesn't contain the answer -- it might still exist \
elsewhere in the video that wasn't retrieved for this question. Say so rather than \
assuming this excerpt is complete.

Other rules:
- When an (audio) line and a (visual) line are close together in time, treat them \
as describing the same moment: reason about what the person is DOING by combining \
what was said with what's shown -- don't just describe the frame in isolation.
- Cite timestamps in [mm:ss] format when your answer relies on a specific moment.

Retrieved context:
{context}
"""


def _format_timestamp(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"


def format_context(items: list[IndexItem]) -> str:
    """Render retrieved items as timestamped, source-tagged lines for the prompt."""
    if not items:
        return "(nothing retrieved)"

    lines = []
    for item in items:
        tag = f"{item.kind}, {item.gap_note}" if item.gap_note else item.kind
        lines.append(f"[{_format_timestamp(item.timestamp)}] ({tag}) {item.text}")
    return "\n".join(lines)


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
