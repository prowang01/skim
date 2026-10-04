# Skim technical notes

Implementation reference and interview preparation. [The README](../README.md)
is the project overview. [The project story](project-story.md) preserves the
personal narrative; [the original implementation log](archive/implementation-log.md)
retains the detailed development account. Historical observations below are labeled
separately from what the current source and saved evaluation snapshot establish.

## Current implementation

### Ingestion and indexing

- `ingest/extract.py`: ffmpeg extracts mono 16 kHz WAV audio.
- `ingest/transcribe.py`: local faster-whisper `small`, CPU/int8, `beam_size=5`.
  The UI passes `None`, `en`, or `fr`; default transcription uses auto-detection.
- `ingest/frames.py`: scene threshold 0.4; attempts an opening frame when no
  detected frame occurs within the first second. If fewer than three frames are
  available, it adds uniform-time samples. A uniform subset caps the returned
  list at 25. Extraction failures can still leave no usable frames.
- `ingest/describe.py`: one GPT-4o vision request for selected images, with
  structured JSON output and instructions to preserve on-screen text. `fr` selects
  French descriptions; `None` and `en` select English. Descriptions can be wrong.
- `index/build_index.py`: embeds transcript segments and nonempty descriptions
  using `text-embedding-3-small` in one batched request. L2-normalized rows in a
  NumPy matrix support cosine search. Items contain kind, timestamp, text, and an
  optional gap note. Empty input returns an empty index without an API call.

The frame cap bounds images sent to vision. ffmpeg still scans the video and writes
all scene candidates before subsampling; transcription and transcript embeddings
also grow with content length. Batching reduces request count, not total token cost.
There are no measured latency/cost results or arbitrary-length guarantees.

The app keeps its index in Streamlit session state. It does not persist a video
library. Its processing key is `(uploaded_file.name, language)`, not a content hash;
same-name replacement uploads can therefore reuse stale session data. This known
prototype limitation was not addressed in the portfolio cleanup.

### Retrieval sequence

`index/retrieve.py` implements:

1. Embed the current question; compare against all indexed rows.
2. Default seeds: `clamp(round(sqrt(n) * 1.4), 6, 40)`, limited by available items.
3. Optional rerank path: retrieve `clamp(round(sqrt(n) * 6), 40, 200)` candidates,
   then keep eight after local `cross-encoder/ms-marco-MiniLM-L-6-v2` scoring.
4. Expand each seed with two preceding and two following same-modality items,
   sorted by timestamp. This is a position window, not a fixed time window.
5. For that expanded set, add opposite-modality items within +/-10 seconds.
6. Target `clamp(round(n * 0.35), 15, n)` total items. Preserve every seed and trim
   expansion-only additions. Extras are retained in chronological order; the cap
   can discard later adjacent evidence and is not optimized by relevance.
7. Annotate items whose nearest opposite-modality timestamp in the full index is
   more than 20 seconds away. Current code leaves the note unset when the other
   modality is completely absent. Notes measure timestamp distance, not full
   interval coverage or whether every relevant frame reached the final context.

The floor means tiny corpora can be retrieved in full. Seeds can exceed the cap.
Neither seed count nor the relative cap enforces a token budget. These heuristics
were explored on a small dataset, not established as generally optimal settings.

### Answer generation and product controls

`qa/answer.py` sends chronological, source-tagged context to `gpt-4o-mini`, with
instructions for timestamp citations, fusion, motion limitations, and uncertainty
when the retrieved excerpt is insufficient. The concise/detailed control changes
the answer instruction, not retrieval breadth. Chat history is passed to the answer
model; retrieval embeds only the current question, without history-based rewriting.
Gap annotations and prompts encourage honesty but cannot enforce it.

`app.py` exposes transcript lines, frame thumbnails/descriptions, and a retrieved
context expander for the newly answered question. The expander shows source text
and timestamps; gap notes are included in the model prompt, not displayed there.

## Retrieval experiments and the rejected reranker

The prior build notes describe a ~32-minute, nine-topic podcast with 911 indexed
items. Debugging identified three different failure modes:

- **Candidate recall:** fixed k=6 excluded answer-bearing passages. Wider adaptive
  selection reached the emotion-before-event explanation, reported at ranks 18-23.
- **Low semantic ranking:** the balloon narrative shared little wording with the
  question about holding on too tightly. A wider rerank candidate pool could include
  it, but inclusion did not ensure final selection.
- **Announcement versus answer:** the introductory line about a powerful analogy
  scored above the actual balloon story. Adding nearby sequence context supplied
  the answer-bearing passage without relying on it ranking independently.

These explanations motivate the current defaults; they do not show that every
instance of these failure modes is solved.

The historical six-video / 15-question experiments reported:

| Retrieval configuration | Podcast score (4 questions) |
| --- | ---: |
| Fixed k=6 | 1.5 / 4 |
| Adaptive top-k | 3.0 / 4 |
| Retrieve then rerank | 2.0 / 4 after manual correction (raw judge: 3.0 / 4) |
| Adjacent-context expansion | 4.0 / 4 in the reported run |

The earlier full-suite comparison reported **11.5/15 with reranking versus 12.0/15
with adaptive top-k**. The log separately records a manual judge correction on a
rerank answer, so these are historical reported figures, not a matched, archived
benchmark pair. Neither rerank artifacts nor the older raw ablations are retained
as separate results files. The larger MiniLM L-12 variant was also reported not to
improve the hardest case. Reranking was rejected as the default because these
experiments did not justify extra model/dependency cost. It remains available for
experimentation, not presented as a proven improvement.

Adjacent expansion was reported to improve the podcast without regressions on the
other five videos in that run. Later runs varied. The current checked-in snapshot
has 4.0/4 on the podcast; it does not prove regression-free behavior in general.

### Context-cap investigation

The log records TED contexts growing to 45-63 of 66 items before the cap and 23
items after it. Observed podcast contexts of roughly 160-180 items were below its
319-item cap. The code returns the context unchanged *when* it is below the cap;
that supports the observed no-op, not an unconditional guarantee for this video,
all queries, or different ingestion output. The cap limits expansion size while
leaving recall/precision trade-offs unresolved.

## Evaluation method and reproducibility

- Dataset: seven videos, 19 handwritten question/expected-answer pairs. Categories
  include factual, cross-modal, reasoning, not-in-video, motion blind spot, and
  needle-in-haystack. Questions categorized cross-modal do not themselves prove
  that both modalities were necessary; there is no modality ablation.
- Baseline: `evals/naive_baseline.py` uses the same ingested index and answer model,
  but formats the entire index rather than a retrieved subset. It has its own
  prompt, including fusion and motion honesty; it lacks retrieval gap notes and
  the same answer-length instruction. This is a pipeline comparison, not a
  controlled isolation of retrieval or blind-spot handling.
- Judge: `gpt-4o-mini` structured output with reasoning then verdict, scored as
  correct = 1, partial = 0.5, wrong = 0. An explicit central-fact rubric aims to
  avoid rewarding a refusal when the expected answer contains a knowable fact.
  Prompt wording and schema order do not guarantee reliable grading.
- Saved snapshot: `evals/results.json` has 38 rows (two systems per question), with
  answers and judge reasoning. Summed scores are 17.5/19 versus 17.0/19. All
  per-video scores match except the podcast (4.0/4 versus 3.5/4).
- Provenance limits: the results file has no run timestamp, commit, retrieval
  configuration, model version snapshot, item counts, or token/cost measurements.
  The prior README identifies it as a default-config run; that attribution cannot
  be independently recovered from the JSON alone. Earlier manual corrections
  belong to the build history, not a retained audit trail in this snapshot.
- French coverage: the harness calls transcription and frame description without
  a language override. It exercises auto-detected transcription and French Q&A,
  with English descriptions by default on fresh ingestion. It does not verify
  the UI's explicit French selection or forced French frame descriptions.
- Cache: ingestion JSON is keyed by video content hash (first 16 hex characters)
  plus file stem and checked against `CACHE_VERSION = 1`. It does not include
  ingestion settings/code/model versions. Cache hits stabilize the stored input;
  they do not make fresh ingestion deterministic. Embeddings, answers, and judging
  run again. `--no-cache` bypasses reads but still refreshes stored ingestion.
- Re-running the full suite overwrites the saved results; `--fast` writes the
  ignored `results_fast.json` for rice, ted, and gelee_groseille. `--clear-cache`
  removes the ingestion cache and exits. Video files are not supplied.

Read answer text and judge reasoning when interpreting scores. The log describes
both borderline correct/partial variation and a clear false-positive grade on a
refusal. No confidence interval, repeated-run aggregate, or independent human
score is available. Citation accuracy is not evaluated separately. The small score
difference does not establish broad superiority, measured cost savings, or
production readiness.

## Known ingestion failures

The pasta case is especially useful for explaining why retrieval cannot repair
missing evidence. Both systems miss the bacon ingredient in the saved snapshot.
The build log reports that a brief "Bacon" overlay produced scene scores around
0.001-0.03, below the 0.4 threshold; the next sampled scene arrived after the label
had disappeared. The raw frame probes are not checked in, so that causal diagnosis
is historical, while the shared wrong answers are preserved in the current JSON.

Near-silent/music-only audio also produced phantom speech in the build notes.
`transcribe.py` does not enable dedicated VAD filtering or retain confidence fields.
Frame descriptions may misread labels or infer ingredients from appearance.
Motion questions cannot be answered reliably from sampled stills alone.

## Interview preparation

- Walk through upload -> transcript/frames -> shared index -> question-specific
  retrieval -> expansion -> gap notes -> answer; point to the UI's inspectable evidence.
- Explain why an announcement can be relevant but insufficient, and why adjacent
  context helped where a cross-encoder did not.
- Explain the reranker rejection with the historical experiment, dependency trade-off,
  and limitations of the retained evidence.
- Distinguish a coverage annotation from a hallucination guarantee, and retrieval
  failure from an ingestion failure.
- Explain why the full-context baseline was competitive and why the half-point
  difference is insufficient to claim a general quality win.
- For another iteration, require new evidence before changing sampling, thresholds,
  or retrieval strategy. Hybrid search, VAD, direct-link ingestion, and voting judges
  remain ideas, not implemented features. The [archived roadmap](archive/roadmap.md)
  and [original spec](archive/build-spec.md) preserve those proposals.
