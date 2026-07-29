"""Semantic search over the in-memory index, with temporal alignment so
audio and visual context describing the same moment are fused together
instead of surfaced as disconnected facts."""

import os
from dataclasses import replace

import numpy as np
from openai import OpenAI

from index.build_index import Index, IndexItem, EMBEDDING_MODEL

MIN_TOP_K = 6
MAX_TOP_K = 40
TOP_K_SQRT_FACTOR = 1.4
TIME_WINDOW_SECONDS = 10.0
GAP_FLAG_THRESHOLD_SECONDS = 20.0


def _default_top_k(n_items: int) -> int:
    """Scale k with corpus size instead of a fixed constant. A fixed k=6 was
    tuned on small (dozens-of-items) test videos; on a 911-item, 9-topic
    video it left correct passages ranked as low as #18-23 out of ~900,
    beaten by unrelated segments that merely share surface vocabulary with
    the question. sqrt growth raises k for larger corpora while tapering off
    (a 10,000-item video doesn't get k=350) -- clamped to [MIN_TOP_K,
    MAX_TOP_K] so tiny videos keep today's tested behavior and huge ones
    stay cost-bounded rather than drifting back toward a full dump."""
    return int(np.clip(round((n_items**0.5) * TOP_K_SQRT_FACTOR), MIN_TOP_K, MAX_TOP_K))


def _embed_query(question: str) -> np.ndarray:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[question])
    vector = np.array(response.data[0].embedding)
    return vector / np.linalg.norm(vector)


def _top_k(index: Index, question: str, k: int) -> list[IndexItem]:
    query_vector = _embed_query(question)
    similarities = index.matrix @ query_vector
    top_indices = np.argsort(-similarities)[: min(k, len(index.items))]
    return [index.items[i] for i in top_indices]


def _expand_with_temporal_neighbors(
    index: Index, selected: list[IndexItem], window: float
) -> list[IndexItem]:
    """For each retrieved item, pull in same-moment context from the other
    modality even if it wasn't independently relevant to the question's
    wording -- e.g. "on screen you can see X" (audio) rarely matches a
    question's phrasing as well as it matches by being right next to the
    frame that actually shows X."""
    selected_keys = {(item.kind, item.timestamp) for item in selected}
    expanded = list(selected)

    for item in selected:
        other_kind = "visual" if item.kind == "audio" else "audio"
        for candidate in index.items:
            key = (candidate.kind, candidate.timestamp)
            if (
                candidate.kind == other_kind
                and key not in selected_keys
                and abs(candidate.timestamp - item.timestamp) <= window
            ):
                selected_keys.add(key)
                expanded.append(candidate)

    expanded.sort(key=lambda i: i.timestamp)
    return expanded


def _nearest_gap(index: Index, item: IndexItem) -> float | None:
    other_kind = "visual" if item.kind == "audio" else "audio"
    gaps = [abs(c.timestamp - item.timestamp) for c in index.items if c.kind == other_kind]
    return min(gaps) if gaps else None


def _annotate_gaps(index: Index, items: list[IndexItem], threshold: float) -> list[IndexItem]:
    """Flag items with no nearby context from the other modality, so the LLM
    has a computed signal for "this moment might be a blind spot" instead of
    having to judge raw timestamp gaps itself (which it's bad at)."""
    annotated = []
    for item in items:
        gap = _nearest_gap(index, item)
        other_kind = "visual" if item.kind == "audio" else "audio"
        note = f"no {other_kind} context within {gap:.0f}s" if gap is not None and gap > threshold else None
        annotated.append(replace(item, gap_note=note))
    return annotated


def retrieve(
    index: Index,
    question: str,
    k: int | None = None,
    window: float = TIME_WINDOW_SECONDS,
    gap_threshold: float = GAP_FLAG_THRESHOLD_SECONDS,
) -> list[IndexItem]:
    """Retrieve the most relevant chunks for a question, temporally align
    them with same-moment context from the other modality, and flag any
    that remain isolated (no nearby context at all) as possible blind spots.

    k defaults to a corpus-size-adaptive value (see _default_top_k) --
    pass an explicit k to override."""
    if not index.items:
        return []
    if k is None:
        k = _default_top_k(len(index.items))
    top = _top_k(index, question, k)
    expanded = _expand_with_temporal_neighbors(index, top, window)
    return _annotate_gaps(index, expanded, gap_threshold)
