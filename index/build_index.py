"""In-memory semantic index over transcript segments + frame descriptions.

Both sources are embedded in a single batched call and kept as a plain numpy
matrix -- at the scale of one video (dozens of segments/frames) a vector DB
or FAISS would be pure overhead."""

import os
from dataclasses import dataclass

import numpy as np
from openai import OpenAI

from ingest.transcribe import Segment
from ingest.describe import FrameDescription

EMBEDDING_MODEL = "text-embedding-3-small"


@dataclass
class IndexItem:
    kind: str  # "audio" or "visual"
    timestamp: float
    text: str


@dataclass
class Index:
    items: list[IndexItem]
    matrix: np.ndarray  # (n, dim), L2-normalized rows, for cosine similarity via dot product


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def build_index(
    segments: list[Segment],
    frame_descriptions: list[FrameDescription],
) -> Index:
    items = [IndexItem(kind="audio", timestamp=s.start, text=s.text) for s in segments]
    items += [
        IndexItem(kind="visual", timestamp=f.timestamp, text=f.description)
        for f in frame_descriptions
        if f.description
    ]

    if not items:
        return Index(items=[], matrix=np.zeros((0, 0)))

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[item.text for item in items])
    vectors = np.array([d.embedding for d in response.data])

    return Index(items=items, matrix=_normalize(vectors))
