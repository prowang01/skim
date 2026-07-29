"""Semantic search over the in-memory index, with temporal alignment so
audio and visual context describing the same moment are fused together
instead of surfaced as disconnected facts."""

import os
from dataclasses import replace

import numpy as np
from openai import OpenAI

from index.build_index import Index, IndexItem, EMBEDDING_MODEL
from index.rerank import rerank

# Stage 1 (bi-encoder): cast a wide net. Sized from real rank data, not the
# "40-50" ballpark that would be reasonable for a smaller failure -- on the
# 911-item podcast eval video, one correct passage ranked #129 by cosine
# similarity alone. A flat top-50 candidate pool would never even contain
# it, and a cross-encoder can only re-rank candidates it receives -- it
# can't rescue something Stage 1 never fetched. Running locally (no API
# cost), a wider net only costs a bit of CPU time, so we can afford to scale
# further than the final top-k ever could.
CANDIDATE_MIN_K = 40
CANDIDATE_MAX_K = 200
CANDIDATE_SQRT_FACTOR = 6.0

# Stage 2 (cross-encoder): once precisely re-scored, a small fixed number is
# enough -- the point of reranking is that we no longer need to over-fetch
# to compensate for cosine similarity's imprecision.
FINAL_RERANK_K = 8

TIME_WINDOW_SECONDS = 10.0
GAP_FLAG_THRESHOLD_SECONDS = 20.0


def _default_candidate_k(n_items: int) -> int:
    """Scale the Stage 1 candidate pool with corpus size. sqrt growth raises
    it for larger corpora while tapering off (a 10,000-item video doesn't
    get a 2,000-item candidate pool) -- clamped to [CANDIDATE_MIN_K,
    CANDIDATE_MAX_K] so tiny videos don't over-fetch and huge ones stay
    bounded well short of "most of the corpus", which would erode the whole
    point of narrowing before the (still real, if free) cost of reranking."""
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
    """Two-stage retrieval: a bi-encoder (embedding cosine similarity) casts
    a wide net of candidate_k items, then a local cross-encoder reranks
    those candidates against the question and keeps the best final_k. The
    winners are then temporally aligned with same-moment context from the
    other modality, and flagged if isolated (no nearby context at all) as
    possible blind spots.

    candidate_k defaults to a corpus-size-adaptive value (see
    _default_candidate_k) -- pass an explicit value to override."""
    if not index.items:
        return []
    if candidate_k is None:
        candidate_k = _default_candidate_k(len(index.items))
    candidates = _top_k(index, question, candidate_k)
    top = rerank(question, candidates, final_k)
    expanded = _expand_with_temporal_neighbors(index, top, window)
    return _annotate_gaps(index, expanded, gap_threshold)
