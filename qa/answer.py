"""Chat over a video's content: fuses retrieved transcript segments and
frame descriptions, temporally aligned, so the model reasons about actions
(what's said + what's shown together) rather than describing frames in
isolation."""

import os

from openai import OpenAI

from index.build_index import IndexItem
from utils import format_timestamp

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
- {length_instruction}

Retrieved context:
{context}
"""

CONCISE_INSTRUCTION = (
    "Answer concisely: 1-2 sentences giving the key fact and its [mm:ss] "
    "citation, no extra elaboration. The honesty/blind-spot rules above still "
    "come first -- if the excerpt doesn't have the answer, say so plainly and "
    "briefly rather than guessing just to sound complete."
)
DETAILED_INSTRUCTION = (
    "Answer thoroughly: give the key fact and its [mm:ss] citation, plus "
    "surrounding context and a brief explanation of how you got there. The "
    "honesty/blind-spot rules above still apply -- more detail doesn't mean "
    "filling in what the excerpt doesn't actually say."
)
DEPTH_INSTRUCTIONS = {"concise": CONCISE_INSTRUCTION, "detailed": DETAILED_INSTRUCTION}


def format_context(items: list[IndexItem]) -> str:
    """Render retrieved items as timestamped, source-tagged lines for the prompt."""
    if not items:
        return "(nothing retrieved)"

    lines = []
    for item in items:
        tag = f"{item.kind}, {item.gap_note}" if item.gap_note else item.kind
        lines.append(f"[{format_timestamp(item.timestamp)}] ({tag}) {item.text}")
    return "\n".join(lines)


def answer_question(
    items: list[IndexItem],
    question: str,
    history: list[dict] | None = None,
    depth: str = "concise",
) -> str:
    """Answer a question given the already-retrieved, temporally-aligned
    context items (see index/retrieve.py). depth is "concise" or "detailed"
    -- it only changes the answer's length/elaboration, never retrieval."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    context = format_context(items)
    length_instruction = DEPTH_INSTRUCTIONS.get(depth, CONCISE_INSTRUCTION)
    system_prompt = SYSTEM_PROMPT.format(context=context, length_instruction=length_instruction)
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(model=CHAT_MODEL, messages=messages)
    return response.choices[0].message.content
