"""Semantic search over the in-memory index, with temporal alignment so
audio and visual context describing the same moment are fused together
instead of surfaced as disconnected facts."""

import os
from dataclasses import replace

import numpy as np
from openai import OpenAI

from index.build_index import Index, IndexItem, EMBEDDING_MODEL
from index.rerank import rerank

# Cross-encoder rerank is opt-in: it showed no net improvement over plain
# adaptive top-k in eval testing, so it stays off unless requested (see
# README "Design decisions").
RERANK_ENV_VAR = "SKIM_ENABLE_RERANK"

# Default path: single-stage adaptive top-k. sqrt growth with a floor/ceiling
# keeps k small for short videos and bounded for long ones (see README for
# how these bounds were chosen from real rank data).
MIN_TOP_K = 6
MAX_TOP_K = 40
TOP_K_SQRT_FACTOR = 1.4

# Opt-in path: Stage 1 (bi-encoder) candidate pool, wider than the default
# top-k since Stage 2 needs a correct passage to be in the pool at all
# before it can rerank it to the top.
CANDIDATE_MIN_K = 40
CANDIDATE_MAX_K = 200
CANDIDATE_SQRT_FACTOR = 6.0

# Stage 2 (cross-encoder) keeps a small fixed number -- precise re-scoring
# means we no longer need to over-fetch to compensate for cosine similarity.
FINAL_RERANK_K = 8

TIME_WINDOW_SECONDS = 10.0
GAP_FLAG_THRESHOLD_SECONDS = 20.0

# Same-modality neighbors to pull in around each selected item (a position
# window, not a time window) -- catches answers that sit a few segments
# after a passage that merely announces them. Kept small: each neighbor also
# gets its own cross-modal expansion downstream, so context size compounds
# quickly with this number.
ADJACENT_NEIGHBOR_COUNT = 2


def _rerank_enabled() -> bool:
    return os.environ.get(RERANK_ENV_VAR, "false").strip().lower() in ("1", "true", "yes")


def _default_top_k(n_items: int) -> int:
    """Adaptive top-k for the default (rerank-disabled) path."""
    return int(np.clip(round((n_items**0.5) * TOP_K_SQRT_FACTOR), MIN_TOP_K, MAX_TOP_K))


def _default_candidate_k(n_items: int) -> int:
    """Adaptive Stage 1 candidate pool size for the rerank-enabled path."""
    return int(np.clip(round((n_items**0.5) * CANDIDATE_SQRT_FACTOR), CANDIDATE_MIN_K, CANDIDATE_MAX_K))


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


def _expand_with_adjacent_context(
    index: Index, selected: list[IndexItem], neighbor_count: int
) -> list[IndexItem]:
    """Pull in neighbor_count same-modality chunks immediately before and
    after each selected item, by position in the transcript/frame sequence
    (not by time). Catches answers that sit a few segments after a passage
    that only announces them -- see README for why this exists."""
    if neighbor_count <= 0:
        return selected

    by_kind: dict[str, list[IndexItem]] = {}
    for item in index.items:
        by_kind.setdefault(item.kind, []).append(item)
    for items in by_kind.values():
        items.sort(key=lambda i: i.timestamp)

    selected_keys = {(item.kind, item.timestamp) for item in selected}
    expanded = list(selected)

    for item in selected:
        sequence = by_kind.get(item.kind, [])
        try:
            pos = next(i for i, candidate in enumerate(sequence) if candidate.timestamp == item.timestamp)
        except StopIteration:
            continue
        lo = max(0, pos - neighbor_count)
        hi = min(len(sequence), pos + neighbor_count + 1)
        for neighbor in sequence[lo:hi]:
            key = (neighbor.kind, neighbor.timestamp)
            if key not in selected_keys:
                selected_keys.add(key)
                expanded.append(neighbor)

    expanded.sort(key=lambda i: i.timestamp)
    return expanded


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
    final_k: int = FINAL_RERANK_K,
    candidate_k: int | None = None,
    window: float = TIME_WINDOW_SECONDS,
    gap_threshold: float = GAP_FLAG_THRESHOLD_SECONDS,
) -> list[IndexItem]:
    """Retrieve the most relevant chunks for a question, expand each with
    its same-modality neighbors and same-moment context from the other
    modality, and flag any that remain isolated as possible blind spots.

    Default: single-stage adaptive top-k. If SKIM_ENABLE_RERANK is set, uses
    two-stage retrieval instead -- a wide bi-encoder candidate pool narrowed
    by a local cross-encoder (see README for why this isn't on by default)."""
    if not index.items:
        return []

    if _rerank_enabled():
        if candidate_k is None:
            candidate_k = _default_candidate_k(len(index.items))
        candidates = _top_k(index, question, candidate_k)
        top = rerank(question, candidates, final_k)
    else:
        top = _top_k(index, question, _default_top_k(len(index.items)))

    top = _expand_with_adjacent_context(index, top, ADJACENT_NEIGHBOR_COUNT)
    expanded = _expand_with_temporal_neighbors(index, top, window)
    return _annotate_gaps(index, expanded, gap_threshold)
