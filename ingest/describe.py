"""Frame description via GPT-4o vision, one batched call for all frames."""

import base64
import json
import os
from dataclasses import dataclass

from openai import OpenAI

from ingest.frames import Frame

VISION_MODEL = "gpt-4o"

SYSTEM_PROMPT = """You are given several still frames extracted from a video, in \
chronological order, each labeled with its index. Describe concisely what each \
frame shows: objects, setting, on-screen text or data (transcribe exact text, \
numbers, and figures verbatim if visible -- this is often the only place such \
data appears), and any notable action captured in that instant. Keep each \
description to 1-3 sentences. Return exactly one entry per frame, using the \
given index."""

FRAME_DESCRIPTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "frames": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "description": {"type": "string"},
                },
                "required": ["index", "description"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["frames"],
    "additionalProperties": False,
}


@dataclass
class FrameDescription:
    timestamp: float
    description: str


def _encode_image(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def describe_frames(frames: list[Frame]) -> list[FrameDescription]:
    """Describe all frames in a single GPT-4o call instead of one call per
    frame -- avoids paying the fixed per-request overhead (system prompt,
    latency) up to 25 times over."""
    if not frames:
        return []

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    content = [{"type": "text", "text": f"Here are {len(frames)} frames from a video, in order."}]
    for i, frame in enumerate(frames):
        content.append({"type": "text", "text": f"Frame index {i}:"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _encode_image(frame.path), "detail": "high"},
            }
        )

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "frame_descriptions",
                "strict": True,
                "schema": FRAME_DESCRIPTIONS_SCHEMA,
            },
        },
    )

    parsed = json.loads(response.choices[0].message.content)
    by_index = {item["index"]: item["description"] for item in parsed["frames"]}

    return [
        FrameDescription(timestamp=frame.timestamp, description=by_index.get(i, ""))
        for i, frame in enumerate(frames)
    ]
