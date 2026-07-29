"""Semantic search over the in-memory index, with temporal alignment so
audio and visual context describing the same moment are fused together
instead of surfaced as disconnected facts."""

import os
from dataclasses import replace

import numpy as np
from openai import OpenAI

from index.build_index import Index, IndexItem, EMBEDDING_MODEL
from index.rerank import rerank

# Cross-encoder rerank is opt-in (see _rerank_enabled) -- the eval suite
# showed no net improvement over plain adaptive top-k (11.5/15 vs 12.0/15)
# and it pulls in a heavy torch/sentence-transformers dependency, so it
# defaults OFF. Set SKIM_ENABLE_RERANK=true to turn it on. See README
# "Design decisions" and "Eval results" for the full before/after.
RERANK_ENV_VAR = "SKIM_ENABLE_RERANK"

# Default path (rerank disabled): a single adaptive top-k, the Palier 4
# fix. Tuned on real rank data -- a fixed k=6 left correct passages ranked
# as low as #18-23 out of ~900 on a large multi-topic video.
MIN_TOP_K = 6
MAX_TOP_K = 40
TOP_K_SQRT_FACTOR = 1.4

# Opt-in path (rerank enabled): Stage 1 (bi-encoder) casts a wider net,
# sized from real rank data -- on the 911-item podcast eval video, one
# correct passage ranked #129 by cosine similarity alone, and a cross-
# encoder can only re-rank candidates it receives. Running locally (no API
# cost), a wider net only costs a bit of CPU time.
CANDIDATE_MIN_K = 40
CANDIDATE_MAX_K = 200
CANDIDATE_SQRT_FACTOR = 6.0

# Stage 2 (cross-encoder): once precisely re-scored, a small fixed number is
# enough -- the point of reranking is that we no longer need to over-fetch
# to compensate for cosine similarity's imprecision.
FINAL_RERANK_K = 8

TIME_WINDOW_SECONDS = 10.0
GAP_FLAG_THRESHOLD_SECONDS = 20.0

# Adjacent-context (sentence-window) expansion: for each selected item, pull
# in this many same-modality neighbors immediately before/after it in the
# transcript/frame sequence -- not a time window, a position window. Targets
# passages where the highest-ranked chunk *announces* an answer ("he shares
# a powerful analogy...") but the actual content is a few segments away.
# Kept conservative (not 3+): each additional neighbor also gets its own
# cross-modal expansion downstream, so context grows fast with window size,
# and with it the risk of diluting precision with less-relevant filler.
ADJACENT_NEIGHBOR_COUNT = 2


def _rerank_enabled() -> bool:
    return os.environ.get(RERANK_ENV_VAR, "false").strip().lower() in ("1", "true", "yes")


def _default_top_k(n_items: int) -> int:
    """Scale k with corpus size instead of a fixed constant (rerank-disabled
    default path)."""
    return int(np.clip(round((n_items**0.5) * TOP_K_SQRT_FACTOR), MIN_TOP_K, MAX_TOP_K))


def _default_candidate_k(n_items: int) -> int:
    """Scale the Stage 1 candidate pool with corpus size (rerank-enabled
    path). sqrt growth raises it for larger corpora while tapering off (a
    10,000-item video doesn't get a 2,000-item candidate pool) -- clamped to
    [CANDIDATE_MIN_K, CANDIDATE_MAX_K] so tiny videos don't over-fetch and
    huge ones stay bounded well short of "most of the corpus", which would
    erode the whole point of narrowing before the (still real, if free)
    cost of reranking."""
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
    """Pull in the neighbor_count same-modality chunks immediately before and
    after each selected item, in transcript/frame order -- a position
    window, not a time window. Targets passages where the highest-ranked
    chunk *announces* an answer but the actual content is a few segments
    away, which neither cosine similarity nor a passage-relevance
    cross-encoder reliably retrieves on its own (see README)."""
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
            pos = next(i for i, it in enumerate(sequence) if it.timestamp == item.timestamp)
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
    modality, and flag any that remain isolated (no nearby context at all)
    as possible blind spots.

    Default (SKIM_ENABLE_RERANK unset/false): a single adaptive top-k over
    the bi-encoder ranking (see _default_top_k) -- the proven Palier 4 path.

    Opt-in (SKIM_ENABLE_RERANK=true): two-stage retrieval -- the bi-encoder
    casts a wide net of candidate_k items (see _default_candidate_k), then a
    local cross-encoder reranks those candidates and keeps the best
    final_k. Not on by default; see README for why."""
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
