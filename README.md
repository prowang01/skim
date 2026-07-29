# Skim

Chat with a video's spoken *and* visual content. Upload an `.mp4`, get a timestamped
transcript plus descriptions of key visual frames, and ask questions in a chat box —
each question retrieves the relevant moments (not the whole video) and answers cite
the `[mm:ss]` timestamps they're based on, whether that's something said or shown.
The system also knows its own blind spots: it says so when a question needs
continuous motion it can't see, or falls in a gap between what was sampled.

**Scope decision:** built for talking-style videos — tutorials, talks, interviews,
lectures — where the meaning lives mostly in audio plus a few key visuals. Not built
for fast-motion or sports footage (see Limitations and Eval results below for exactly
how it handles that case when asked anyway).

This is **Palier 4** of a larger plan (see `videolens-spec.md`): blind-spot awareness
and evals. Palier 1 was audio-only; Palier 2 added visual descriptions but dumped
everything into every question; Palier 3 added retrieval + temporal fusion. Palier 4
adds explicit honesty about what the system can't reliably answer, plus a small eval
harness comparing this system against the Palier-1/2-style naive baseline. Each palier
is meant to be a complete, working project on its own — this one is it for now.

## How it works

1. **ffmpeg** extracts mono 16kHz audio from the uploaded video (`ingest/extract.py`).
2. **faster-whisper** (local, CPU, `small` model) transcribes it into timestamped
   segments (`ingest/transcribe.py`).
3. **ffmpeg scene detection** (`select='gt(scene,0.4)'`) captures a frame only when
   the image changes significantly, hard-capped at ~25 frames sampled uniformly
   across all detected changes if there are more. The video's opening frame is
   always included too, and a uniform-time-sampling fallback fills in if too few
   scene changes are detected, so the visual index is never empty
   (`ingest/frames.py`).
4. **GPT-4o** describes all sampled frames in a single batched vision call —
   transcribing any on-screen text/numbers verbatim — via structured output
   (`ingest/describe.py`).
5. **Indexing:** every transcript segment and frame description is embedded with
   `text-embedding-3-small` in one batched call and kept as an in-memory numpy matrix
   of normalized vectors, alongside a parallel list of `{kind, timestamp, text}`
   (`index/build_index.py`).
6. **Retrieval:** each question is embedded and compared by cosine similarity against
   the index to get the top-k most relevant chunks — audio and visual mixed
   together, ranked purely by relevance. Each retrieved chunk is then *expanded*
   with any chunk of the *other* modality within a ±10s time window, even if that
   chunk alone wouldn't have matched the question's wording. Each item is also
   checked for isolation: if the nearest chunk of the *other* modality is more than
   ~20s away, the item is flagged (e.g. "no visual context within 45s") so the model
   has a computed signal for a possible blind spot instead of having to judge raw
   timestamp gaps itself (`index/retrieve.py`).
7. The retrieved, temporally-aligned, gap-flagged items are sorted chronologically
   and placed in an LLM's system prompt, which explicitly instructs it to: treat
   close-in-time audio/visual lines as the same moment and infer what's being *done*;
   cite `[mm:ss]` timestamps; and know its blind spots -- say so plainly rather than
   guess when a question needs motion/speed a still can't show, when an item is
   gap-flagged, or when the retrieved excerpt (not the full video) just doesn't have
   the answer (`qa/answer.py`).
8. Streamlit (`app.py`) wires this into an upload → transcript + frames → chat UI,
   with an expander showing exactly what was retrieved for each question.
9. **Evals** (`evals/`): a hand-built dataset of 5 videos and ~11 questions spanning
   factual recall, cross-modal fusion, reasoning, "not in video" honesty, and motion
   blind-spot honesty. `run_evals.py` runs every question through both this system
   and a naive dump-everything baseline (`evals/naive_baseline.py` -- the Palier-1/2
   approach), grades both with a gpt-4o-mini judge, and prints a side-by-side score.

## Design decisions

- **faster-whisper over the OpenAI Whisper API.** Runs locally on CPU (built/tested
  on an Apple M4 Pro), so there's no per-minute transcription cost — only the
  vision/embedding/chat steps need `OPENAI_API_KEY`.
- **Scene-detection sampling, not 1 frame/sec.** A hard cap on frames, not a hard cap
  on video duration, bounds cost/latency regardless of length.
- **One batched vision call, not one per frame; one batched embedding call, not one
  per chunk.** Both the frame descriptions and the index embeddings are computed in
  a single API call per video — avoids paying fixed per-request overhead dozens of
  times over, and keeps cost negligible even as a video grows.
- **Plain numpy over FAISS/a vector DB.** At the scale of one video (dozens of
  chunks), a `(n, 1536)` matrix and a single matrix-vector dot product for cosine
  similarity is simpler and just as fast as any dedicated index — no DB rabbit hole.
- **Retrieval, not dump.** Only the top-k chunks relevant to a specific question go
  into the prompt, not the entire transcript and all frame descriptions. This is
  what stops the design from breaking down on longer videos, and it's the
  difference between "a script" and a RAG system.
- **Retrieve-then-expand for fusion, not retrieve-then-dump.** Ranking audio and
  visual chunks purely by semantic similarity to the question tends to miss frame
  descriptions that don't share the question's vocabulary (e.g. "shows 4 egg yolks
  on screen" vs. a question like "what are the exact quantities") even when they're
  exactly the moment being asked about. Expanding each retrieved chunk with
  same-time-window chunks from the other modality fixes this without expanding the
  whole context — it's targeted, not a second dump.
- **A computed gap signal, not a prompt-only motion caveat.** Telling the model
  "be honest about stills vs. motion" is cheap prompt wording and works for motion
  questions -- but judging whether a *specific* timestamp gap is "big" from a wall
  of numbers is a different, harder ask an LLM is unreliable at. So gaps (no
  same-item context from the other modality within ~20s) are computed in code and
  annotated directly onto the context, giving the model an objective fact to point
  to ("no visual context within 45s") instead of a judgment call to get wrong.
- **The naive baseline reuses the real pipeline, not a re-implementation.** The eval
  baseline isn't a separate mocked-up system -- it's the same ingested transcript +
  frame descriptions, just handed to the LLM whole instead of retrieved, with the
  Palier-2-era prompt (no retrieval-excerpt caveat, no gap flags). This isolates
  exactly what Palier 3+4 added, rather than comparing against a strawman.
- **LLM-as-judge over exact-match scoring.** Expected answers are hand-written
  ground truth, not exact strings the system should reproduce -- a judge model
  (gpt-4o-mini, forced structured output) grades correct/partial/wrong based on
  whether the substance matches, which handles paraphrasing and "correctly declined
  to answer" cases that string matching can't.

## Eval results

Ran against 6 real videos: 5 short (~3 min) clips spanning a TED talk, a cross-modal
cooking video, a visual-only (near-silent) cooking video, an animated explainer, and
Olympic basketball highlights (fast-motion, the deliberate blind-spot stress test) --
plus one ~32 minute multi-topic compilation (9 different named experts, each on a
different narrow relationship topic) added specifically to stress-test retrieval at
scale with needle-in-haystack questions. 15 questions total across factual recall,
cross-modal fusion, reasoning, not-in-video honesty, motion blind-spot honesty, and
needle-in-haystack recall. Full detail in `evals/results.json`.

```
video                     retrieval          naive
basket_france_usa             3.0/3          3.0/3
pasta                         1.0/2          1.0/2
podcast                       1.5/4          4.0/4
rice                          2.0/2          2.0/2
stock_exchange                1.5/2          1.5/2
ted                           2.0/2          2.0/2
--------------------------------------------------
TOTAL                       11.0/15        13.5/15

category                  retrieval          naive
blind_spot_motion             2.0/2          2.0/2
cross_modal                   1.5/2          1.5/2
factual                       2.0/3          2.0/3
needle_haystack               1.5/4          4.0/4
not_in_video                  2.0/2          2.0/2
reasoning                     2.0/2          2.0/2
--------------------------------------------------
TOTAL                       11.0/15        13.5/15
```

**On the 5 short clips, the two systems tied (9.5/11 each)** -- small enough corpora
(50-83 indexed items) that dumping everything never overwhelmed the naive baseline.
Both blind-spot questions (fast break speed, specific dribble move) were answered
honestly by both systems -- Part A's blind-spot prompt logic holds up. Both missed the
same pasta-sauce question, and the eval process itself surfaced why (see Limitations).

**On the long podcast (911 indexed items), the naive baseline clearly won (4.0/4 vs.
1.5/4) -- the opposite of what retrieval was built to demonstrate.** This deserved a
real root-cause dig rather than being reported at face value. Inspecting exactly what
`index/retrieve.py` surfaced for the two worst-scoring questions:

- *"What analogy does the comedian... use to explain holding on too tightly?"* --
  the actual balloon-analogy passage (~25:20-25:34) was never retrieved. Instead,
  top-k=6 surfaced lines like *"Why are you holding on?"* [01:19] and *"what am I
  really attaching myself to?"* [04:14] -- both from a **different expert's**
  segment on a different topic, pulled in purely because they share surface
  vocabulary ("holding on", "attaching") with the question's wording.
- *"What does the neuroscientist say about... emotion and events?"* -- the actual
  explanation (~16:16-16:36) was also never retrieved. Instead it surfaced that
  expert's segment *intro* line and, again, intro lines from unrelated segments
  (Matthew Hussey at [18:11], James Corden's *"powerful analogy about love and
  letting go"* at [24:55]) -- all lexically adjacent to "manifest love" without
  being the answer.

**Root cause: a fixed top-k=6 doesn't scale to a large, multi-topic corpus.** With
~900 short (~2s) transcript fragments spanning 9 different topics, several
completely unrelated segments happen to share surface-level relationship vocabulary
("holding on", "manifesting", "attaching") with a given question's phrasing. Cosine
similarity ranks those lexical near-misses above the actual answer whenever the real
passage phrases the idea differently than the question does -- exactly what happened
here. The ±10s temporal-alignment expansion doesn't help recall in an audio-only case
like this (frames are sparse, ~1 every 78s, so there's rarely a nearby one to expand
with). The naive baseline, by contrast, can't miss the passage -- it sees the entire
transcript -- and gpt-4o-mini's long-context recall is clearly good enough to locate
a distinctive quote among ~900 lines once it's all there to read. This is a genuine,
now eval-confirmed limitation of the current fixed-k design, not a wash: retrieval's
fusion/honesty logic is sound (it never fabricated -- it just said the excerpt lacked
detail), but recall at this corpus size needs a larger or adaptive k, or a rerank
step, to actually beat a full dump. Left as a documented, prioritized limitation
rather than patched under eval pressure.

## Limitations

- **Fixed top-k=6 doesn't scale to a large, multi-topic corpus -- confirmed by eval,
  not just theorized.** On the ~32min/911-item podcast eval video, retrieval scored
  1.5/4 on needle-in-haystack questions against a naive full-dump baseline's 4/4 (see
  Eval results above for the full root-cause dig). Short, generic-vocabulary segments
  from unrelated topics can out-rank the actual answer in cosine similarity whenever
  the real passage phrases things differently than the question does. Not adapted to
  video length or content density at all currently -- a 3-minute clip and a 32-minute
  compilation get the same k=6 and the same ±10s alignment window.
- **Scene-change detection can structurally miss on-screen text that doesn't
  accompany a shot change.** Investigated a concrete failure during evals: the
  `pasta` video shows a "Bacon" on-screen text label for a full ~3 seconds
  (0.65s-3.65s, confirmed by manual frame probing), clearly long enough to sample.
  But ffmpeg's per-frame `scene_score` throughout that window measured only
  0.001-0.03 -- two orders of magnitude below the 0.4 threshold -- because the shot
  itself barely changes (same board, same hands, same background); only a small
  corner text overlay turns on. Our next sampled frame was 20 seconds later, by
  which point the label was long gone, so neither GPT-4o vision nor the transcript
  ever saw the word "bacon" (vision guessed "minced meat" from the raw meat's
  appearance instead), and both the retrieval system and the naive baseline gave the
  same wrong answer -- garbage in, garbage out, not a fusion or reasoning failure.
  This is a real, more precise version of "frame sampling may miss things between
  samples": it's not just about sampling *density*, pixel-difference scene
  detection is *structurally* blind to static text overlays regardless of how long
  they're shown. Possible fix, not yet implemented: complement scene detection with
  either lightweight periodic time-based sampling (e.g. one frame every N seconds
  regardless of scene score) or on-screen text/caption detection (e.g. OCR-based
  triggering) so a several-second-long label isn't dependent on a shot change to be
  caught.
- Local transcription runs on CPU; a long video will take a while on first upload.
  Test with a short (2-3 min) clip first.
- The `small` faster-whisper model trades some accuracy for speed — expect
  occasional mis-transcriptions, especially with accents, cross-talk, or background
  noise. Whisper's language auto-detection can also occasionally mis-fire on short
  or unusual-sounding audio. On near-silent/music-only audio it can also outright
  hallucinate text (observed on the `pasta` eval video: real output included
  "music" followed by seven repeated "Oh." segments with no corresponding speech) --
  the system has no dedicated detection for this yet, it relies on the LLM
  recognizing repeated/nonsensical transcript lines as unreliable.
- Frame descriptions are only as good as GPT-4o's read of the image — small or
  low-contrast on-screen text may still be misread.
- The eval set is still small (6 videos, 15 questions). It's now large enough to have
  surfaced a real retrieval-at-scale gap (see above), but one long video is one data
  point, not a robust curve of "how does k need to scale with corpus size" -- more
  long/multi-topic videos would sharpen that picture.

## Future work

Not implemented, just noted for later:

- **Adjustable answer depth.** Right now every answer is whatever length the LLM
  defaults to. Letting the user pick concise vs. detailed (e.g. a toggle or a
  system-prompt parameter) would help match the answer to the question -- a quick
  factual lookup and "walk me through the whole segment" shouldn't get the same
  amount of prose.
- **A "go deeper" follow-up action.** After a concise answer, offer a one-click way
  to expand it -- e.g. re-retrieve with a larger k/window scoped to the same
  timestamp range and re-answer with more detail/context, rather than making the
  user rephrase the question to get more.

## Run locally

Requires `ffmpeg` on your `PATH` and Python 3.11.

```bash
pyenv local 3.11.14        # or any Python 3.11
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env       # then fill in OPENAI_API_KEY

streamlit run app.py
```

The first transcription downloads the `small` faster-whisper model (~500MB) to your
local Hugging Face cache; subsequent runs reuse it.

To run the eval suite (needs your own video files under `evals/videos/` matching
`evals/dataset.json`; videos are gitignored and not included in this repo):

```bash
python -m evals.run_evals
```
