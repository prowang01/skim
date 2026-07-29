# Roadmap

Planning doc only — nothing here is implemented yet. Written at the end of a Palier 5
session (retrieval + rerank + adjacent-context + relative cap, see README for the
full story) to organize what's next for a focused 1-2 hour session.

Two buckets: **Quick wins** (minutes each, low risk, do these first) and **Bigger
work** (real chunks — plan for one of these per session, not several).

## Quick wins, ranked by effort-to-value ratio

1. **Fast eval subset** (~15-20 min). `evals/run_evals.py` always runs all 6 videos,
   including re-transcribing the 32-min podcast every time (~3.5 min of wall time on
   its own). Add a way to filter which videos run — an env var
   (`EVAL_VIDEOS=ted,rice`) or a CLI arg is enough; just filter `dataset["videos"]`
   by `id` before the loop. Ranked #1 because every other item below gets verified
   by re-running evals, and this cuts that loop from ~10 min to ~1 min. Do this
   first tomorrow, before anything else.
2. **Manual language selector (FR/EN)** (~15-20 min). `ingest/transcribe.py`'s
   `transcribe()` never passes `language=` to faster-whisper, relying entirely on
   auto-detection — which mis-fired on a synthetic TTS test clip earlier this
   project (detected French for English audio). Add a `language` param
   (`None`/`"en"`/`"fr"`) threaded through to `model.transcribe(..., language=...)`,
   plus a selectbox in `app.py` ("Auto / English / French"). Directly addresses "or
   let user pick language" from your own framing — the reliable-auto-detection
   version is the bigger-work variant below.
3. **Judge rubric tightening** (~10-15 min). The balloon question's chronic
   correct/partial flipping (documented at length in the README) comes from the
   judge's own grading looseness, not the system's answers changing. Tighten
   `JUDGE_SYSTEM_PROMPT` in `evals/run_evals.py` with more explicit criteria (e.g.
   "partial only if a concrete named detail is missing, not for extra correct
   context" or "score on covered specific facts, not phrasing/completeness").
   Reduces noise, doesn't eliminate it — a rubric can't fix an inherently
   probabilistic judge, but it narrows the range. Verify with the fast-eval-subset
   tool from #1 by re-running the podcast questions a few times.
4. **Concise / detailed answer toggle** (~15-20 min). Add a radio/selectbox in
   `app.py` ("Concise / Detailed") that swaps a line in `qa/answer.py`'s
   `SYSTEM_PROMPT` (e.g. "answer in 1-2 sentences" vs. "answer thoroughly with
   supporting detail"). The simplest version of the Future Work item already noted
   in the README — the "go deeper" follow-up *action* (re-retrieve with a bigger
   window) is the bigger-work version below.
5. **Whisper low-confidence segment filtering** (~30-45 min). faster-whisper's
   segment objects carry `no_speech_prob` (and `avg_logprob`) that
   `ingest/transcribe.py`'s `transcribe()` currently discards when building
   `Segment` objects. Filtering out segments above a `no_speech_prob` threshold
   (needs a real value found by testing against the pasta video, not guessed) could
   directly suppress the "Oh"/"Yeah" hallucinations on near-silent audio without a
   new dependency. Slower than #1-4 because it needs a threshold tuned against a
   real case, and a check that it doesn't drop real quiet speech. Full VAD (a
   dedicated library) is the bigger-work version if this proves insufficient.
6. **Investigate the startup torch/transformers warnings further** (~15-20 min,
   uncertain payoff). The lazy-import fix (see README) confirmed a genuinely fresh
   venv has zero torch/transformers footprint — the warnings only reappear in an
   existing dev venv where `sentence-transformers` was installed for the rerank
   experiment, because `ingest.transcribe`'s dependency chain (likely
   `huggingface_hub`'s optional framework detection) opportunistically imports
   torch if it's already present on disk. Worth 15-20 min to confirm the exact
   trigger and check if an env var (e.g. `USE_TORCH=0`, set before any imports in
   `app.py`) suppresses it — but this may turn out to be an unfixable artifact of
   this specific dev venv's history rather than a real bug, in which case it's just
   confirmed and documented, not "fixed". Ranked last because the value is
   cosmetic (a fresh install is already unaffected) and the fix isn't guaranteed to
   exist.

## Bigger work

Real chunks — pick one per session, verify with the full eval suite (or the fast
subset from quick win #1 during iteration, full suite before committing), same
rigor as every fix this session.

### Retrieval quality
- **Real short-video selectivity.** The relative cap (~35%) makes short videos
  *less* over-retrieved, but 35% still isn't truly selective — real RAG-style
  selectivity only clearly matters on long videos (podcast retrieves 18-20%
  without even needing the cap). Worth a proper investigation: does a tighter cap
  (e.g. 15-20%) hurt short-video accuracy, or was 35% just a first conservative
  guess? Needs the same rank-data-grounded approach as prior threshold changes,
  not another guess.
- **Refine adjacent-context (sentence-window) retrieval.** This is the actual fix
  for the "announcement vs. content" failure mode (see README) — worth sharpening
  now that it's proven valuable. Candidate: asymmetric expansion (more forward
  neighbors than backward, since analogies/stories are told *after* an intro line,
  rarely before it) instead of the current symmetric ±2.
- **Hybrid search (vector + keyword/BM25).** Cosine similarity misses exact-term
  matches (names, numbers, specific phrases) that a keyword/BM25 score would catch
  directly. Would need a new lightweight scoring path in `index/` and a fusion
  strategy (e.g. reciprocal rank fusion) with the existing embedding search — real
  design work, not a tuning tweak.
- **Explore reranking on more/larger corpora.** Rerank is opt-in and evaluated as a
  wash on this project's one long video (see README). Worth revisiting only with
  more long-video data points — one video's result isn't a curve.

### Audio / transcription
- **Full VAD (voice activity detection) integration.** If quick-win #5's
  confidence filtering isn't enough, a dedicated VAD library (e.g. silero-vad) run
  before Whisper would skip non-speech audio entirely rather than filter
  after-the-fact. New dependency, real integration work in `ingest/`.
- **Reliable multi-language auto-detection.** Beyond quick-win #2's manual
  override: per-segment language detection, handling code-switching within one
  video, and generally hardening auto-detect reliability. Open-ended, no
  guaranteed outcome — the manual override is the practical version to ship first.

### Evals & judging
- **Majority-vote judging.** Call the judge N times (e.g. 3) per answer and take
  the majority verdict, to reduce (not eliminate) the variance quick-win #3's
  rubric tightening only partially addresses. Real cost/latency tradeoff to think
  through (3x judge calls per question), and `evals/run_evals.py` needs the
  aggregation logic.
- **Adaptive frame density by video type.** Sample more frames for visually-dense
  content (e.g. a cooking demo) and fewer for talking-head content, instead of the
  same scene-detection logic for both. Needs a way to classify "video type" first
  (scene-change frequency during an initial pass? speech-to-silence ratio?) —
  a real design question, not a parameter tweak.

### UX / product
- **"Go deeper" follow-up action.** After a concise answer, a button that
  re-retrieves with a larger window scoped to the same timestamp range and
  re-answers with more detail — the fuller version of quick-win #4. Needs new
  retrieval parameterization and UI wiring tied to the previous turn's context.
- **Layout redesign** (chat right, transcript top-left, frames bottom-left).
  A real `st.columns`-based restructure of `app.py`, not a copy change — worth
  budgeting a full session; Streamlit's chat components have some known quirks
  inside column layouts worth testing early rather than discovering late.

### New ingestion sources
- **YouTube-transcript ingestion.** Pull a video's official captions (e.g. via
  `youtube-transcript-api`) instead of downloading + running Whisper. Real open
  design question before coding: with no downloaded video file, there's no source
  for visual frames at all — decide whether this path is audio-only-by-design (a
  new, explicitly scoped mode) or whether the video still gets partially fetched
  just for frame extraction while skipping local transcription. Don't start coding
  until that's decided.
