"""Cross-encoder reranking, local only (no API key). A cross-encoder scores
the question and a candidate together in one pass, which captures relevance
that cosine similarity misses when the right passage phrases things very
differently than the question does -- at the cost of only being cheap
enough to run on a narrowed candidate set, not the whole index."""

from sentence_transformers import CrossEncoder

from index.build_index import IndexItem

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(RERANK_MODEL)
    return _model


def rerank(question: str, candidates: list[IndexItem], top_k: int) -> list[IndexItem]:
    """Re-score candidates against the question and keep the top_k highest."""
    if not candidates:
        return []
    model = _get_model()
    pairs = [(question, item.text) for item in candidates]
    scores = model.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda pair: -pair[0])
    return [item for _, item in ranked[:top_k]]
