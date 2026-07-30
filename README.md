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

This is **Palier 5** of a larger plan (see `videolens-spec.md`): closing the
retrieval-at-scale gap Palier 4's evals found. Palier 1 was audio-only; Palier 2 added
visual descriptions but dumped everything into every question; Palier 3 added
retrieval + temporal fusion; Palier 4 added blind-spot honesty and an eval harness
comparing this system against a naive dump-everything baseline, which then found
retrieval losing badly on a long, multi-topic video. Palier 5 chased that gap through
three fix attempts, evaluating each one honestly rather than assuming it worked:
adaptive top-k (kept -- see Design decisions), cross-encoder reranking (built,
evaluated, found to be a wash, kept only as an opt-in flag -- `SKIM_ENABLE_RERANK=true`
to try it), and adjacent-context retrieval (kept -- closed the remaining gap with zero
regressions on the rest of the eval set). See Eval results for the full story. Each
palier is meant to be a complete, working project on its own — this one is it for now.

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
6. **Retrieval:** by default, a bi-encoder (embedding cosine similarity) picks the
   top-k most relevant chunks -- audio and visual mixed together -- with k adaptive
   to corpus size (`clamp(round(sqrt(n_items) * 1.4), 6, 40)`). **Optionally**
   (`SKIM_ENABLE_RERANK=true`, off by default -- see Design decisions for why), a
   two-stage pipeline replaces this: Stage 1 casts a wider adaptive candidate net
   (`clamp(round(sqrt(n_items) * 6.0), 40, 200)`), then Stage 2, a local
   cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, via `sentence-transformers`
   -- no API key, runs on CPU) re-scores every candidate jointly against the
   question and keeps the best 8 (`index/rerank.py`). Either way, each winning chunk
   is then expanded twice: first with its **2 nearest same-modality neighbors**
   before and after it in the transcript/frame sequence (not a time window -- a
   position window; this is what catches an answer whose content sits a few
   segments after the passage that merely *announces* it), then with any chunk of
   the *other* modality within a ±10s time window, even if that chunk alone
   wouldn't have matched the question's wording. The result is then capped to at
   most ~35% of the corpus (never below the original top-k/reranked selection) --
   on small videos, the two expansion steps above can otherwise compound until
   retrieval covers nearly the whole video, no longer selective. Each item is also
   checked for isolation: if the nearest chunk of the *other* modality is more than
   ~20s away, the item is flagged (e.g. "no visual context within 45s") so the
   model has a computed signal for a possible blind spot instead of having to
   judge raw timestamp gaps itself (`index/retrieve.py`).
7. The retrieved, temporally-aligned, gap-flagged items are sorted chronologically
   and placed in an LLM's system prompt, which explicitly instructs it to: treat
   close-in-time audio/visual lines as the same moment and infer what's being *done*;
   cite `[mm:ss]` timestamps; and know its blind spots -- say so plainly rather than
   guess when a question needs motion/speed a still can't show, when an item is
   gap-flagged, or when the retrieved excerpt (not the full video) just doesn't have
   the answer (`qa/answer.py`).
8. Streamlit (`app.py`) wires this into an upload → transcript + frames → chat UI,
   with an expander showing exactly what was retrieved for each question.
9. **Evals** (`evals/`): a hand-built dataset of 6 videos and 15 questions spanning
   factual recall, cross-modal fusion, reasoning, "not in video" honesty, motion
   blind-spot honesty, and long-video needle-in-haystack recall. `run_evals.py` runs
   every question through both this system and a naive dump-everything baseline
   (`evals/naive_baseline.py` -- the Palier-1/2 approach), grades both with a
   gpt-4o-mini judge, and prints a side-by-side score broken down by video and category.

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
- **Adaptive candidate pool, sqrt-scaled with a floor and ceiling, not a fixed
  constant.** A fixed k=6 was tuned on small test videos and confirmed to fail on a
  real 911-item, 9-topic video (see Eval results): correct passages ranked as low as
  #18-129, beaten by unrelated segments sharing surface vocabulary with the
  question. sqrt growth (`clamp(round(sqrt(n)*6.0), 40, 200)`) raises the Stage 1
  candidate pool for larger corpora while tapering off -- a 10,000-item video
  doesn't get a 2,000-item candidate pool -- so cost stays bounded. The bounds came
  from real rank data, not a guess, and were revised upward once rerank entered the
  picture: a plain "top 40-50" pool (the initially reasonable-sounding range) would
  never have contained a passage ranked #129, and a cross-encoder can only re-rank
  candidates it actually receives -- it can't rescue something Stage 1 never
  fetched. Because Stage 2 runs locally with no API cost, casting a much wider net
  than a pure bi-encoder top-k ever could only costs a bit of CPU time.
- **Cross-encoder rerank, local, no new API dependency.** A bi-encoder embeds the
  question and each chunk *separately* and compares vectors -- fast, but blind to
  interactions between the two texts. A cross-encoder feeds the question and a
  candidate through the model *together*, which is far more precise but too slow to
  run over an entire corpus -- hence two stages: bi-encoder to narrow (cheap, wide),
  cross-encoder to precisely re-score the narrowed set (precise, small).
  `cross-encoder/ms-marco-MiniLM-L-6-v2` was chosen over the larger L-12-v2 variant
  after directly comparing them on the hardest eval case: L-12-v2 showed no
  improvement (see Eval results) while being slower, so there was no reason to pay
  for the bigger model.
- **Rerank is a feature flag, off by default -- shipped code isn't the same as
  proven-better code.** Re-running the full eval suite with reranking enabled
  scored 11.5/15 against 12.0/15 for plain adaptive top-k -- a wash at best, not a
  win, and it adds a heavy `torch`/`sentence-transformers` dependency for that. It
  stays in the codebase (`SKIM_ENABLE_RERANK=true` to try it) because the
  investigation it enabled was valuable -- it's what surfaced the third,
  narrative-structure failure mode in Eval results -- but "instructive to build" and
  "worth defaulting on" are different bars, and this doesn't clear the second one
  yet.
- **Retrieve-then-expand for fusion, not retrieve-then-dump.** Ranking audio and
  visual chunks purely by semantic similarity to the question tends to miss frame
  descriptions that don't share the question's vocabulary (e.g. "shows 4 egg yolks
  on screen" vs. a question like "what are the exact quantities") even when they're
  exactly the moment being asked about. Expanding each retrieved chunk with
  same-time-window chunks from the other modality fixes this without expanding the
  whole context — it's targeted, not a second dump.
- **Adjacent-context (sentence-window) expansion, on by default, always -- not
  gated behind a flag like rerank.** Both cross-encoders in the rerank experiment
  correctly ranked the passage that *announces* an answer above the passage that
  *contains* it (see Eval results) -- a narrative-structure problem no relevance
  ranker fixes. The actual fix is structural, not a smarter ranker: pull in each
  winning chunk's 2 nearest same-modality neighbors by position, regardless of
  vocabulary or time gap. Evaluated exactly like every other change here -- kept
  only after the full suite showed the podcast's needle-in-haystack score jumping
  from 3.0/4 to a clean 4.0/4 (Q3 rescued, verified against the actual answer text)
  with zero regressions on the other 5 videos. Kept conservative at 2 neighbors
  (not 3+): each additional neighbor also gets its own cross-modal expansion
  downstream, so unrestrained growth would dilute precision fast, and the eval
  set's smaller videos are sensitive to that (see Limitations).
- **A relative cap (~35% of the corpus) on the final retrieved set, bounding what
  the two expansion steps above can compound to.** Adaptive top-k already scales
  the *seed* selection down for small corpora, but expansion is proportional, not
  absolute -- on the 66-item TED talk, the seed (11 items) plus its neighbors and
  cross-modal matches ballooned to 45-63 items (68-95% of the video), no longer
  selective, just a slower version of the naive dump. The cap only trims
  expansion-only additions, never the seed itself, so tiny videos don't lose the
  context they need -- and it structurally cannot affect a large corpus like the
  911-item podcast: expansion there naturally lands around 160-180 items, and the
  cap function's very first check (`if len(expanded) <= cap: return expanded`)
  returns the set completely untouched whenever it's already under the threshold
  (319 items here) -- not "unlikely to bind", provably a no-op at every observed
  size for that video.
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

**This is the default-config table** (`SKIM_ENABLE_RERANK` unset -- adaptive top-k +
adjacent-context expansion, both on by default):

```
video                     retrieval          naive
basket_france_usa             3.0/3          3.0/3
pasta                         1.0/2          1.0/2
podcast                       4.0/4          4.0/4
rice                          2.0/2          2.0/2
stock_exchange                1.5/2          1.5/2
ted                           2.0/2          2.0/2
--------------------------------------------------
TOTAL                       13.5/15        13.5/15

category                  retrieval          naive
blind_spot_motion             2.0/2          2.0/2
cross_modal                   1.5/2          1.5/2
factual                       2.0/3          2.0/3
needle_haystack               4.0/4          4.0/4
not_in_video                  2.0/2          2.0/2
reasoning                     2.0/2          2.0/2
--------------------------------------------------
TOTAL                       13.5/15        13.5/15
```

Retrieval matches naive exactly, at parity rather than trailing -- and gets there
with a bounded, targeted context instead of a full dump. This is the end of a
four-stage journey on the podcast's needle-in-haystack score (same 15 questions
throughout the whole eval set; only the podcast score changed across retrieval
changes -- and only after each stage was actually measured, not assumed):

```
fixed k=6 (Palier 4):                1.5/4
adaptive top-k (Palier 4, cont.):     3.0/4
retrieve-then-rerank (evaluated,
  not kept as default):              2.0/4   <- after correcting a judge misgrade (raw: 3.0/4)
adjacent-context (Palier 5, kept):    4.0/4   <- all 4 podcast questions correct, zero regressions
```

**Fixed k=6:** naive clearly won (4.0/4 vs. 1.5/4). Root-caused: the real
balloon-analogy passage (~25:20-25:34) and the real "feel the emotion first"
explanation (~16:16-16:36) were never retrieved at all -- top-6 surfaced only lines
from **other experts' unrelated segments** that happened to share surface
vocabulary with the question.

**Adaptive top-k** raised the podcast score to 3.0/4 by widening k enough to reach
passages that were rankable but excluded (the "manifest love" explanation sat at
rank #18-23; a wider k reached it). The balloon analogy, at rank #129-422, remained
out of reach of any cost-aware k -- noted at the time as a distinct problem: its
wording ("tie it to my wrist") shares almost no vocabulary with the question
("analogy... holding on too tightly").

**Retrieve-then-rerank was built specifically to fix the balloon case, and it
partially did -- Stage 1 now correctly includes the passage (rank #129 is well
within the new 181-item candidate pool for this corpus size), but Stage 2 still
doesn't select it.** Both `ms-marco-MiniLM-L-6-v2` and the larger `L-12-v2` score
the actual balloon narrative *lower* than a nearby meta-commentary line, *"he shares
a powerful analogy about love and letting go"* [24:55] -- confirmed by directly
comparing both models' scores on the same candidates. This is neither a top-k
problem nor a vocabulary-overlap problem: it's a **third, distinct failure mode**.
The line that announces an answer ("he shares a powerful analogy...") is genuinely
the most *topically relevant* passage to the question -- it literally contains the
word "analogy". Both cross-encoders, trained on query-passage relevance (MS-MARCO),
correctly rank it highest by that standard. But topical relevance isn't the same
thing as *where the answer's content lives* -- the actual balloon story is in the
passages that follow, and neither cosine similarity nor a passage-relevance
cross-encoder is built to bridge "this passage announces an answer" to "the answer
is in the next few passages after it." The likely real fix: **adjacent-context /
sentence-window retrieval** -- when a chunk is selected, automatically pull in a
window of the chunks immediately around it (not just same-timestamp cross-modal
chunks, which is what `index/retrieve.py` already did), since narrated
stories/analogies are told in fragments across consecutive Whisper segments after
an intro line, not in one self-contained chunk.

**Adjacent-context retrieval, tried next, closed the gap -- kept because it was
verified, not just plausible.** Pulling in each winning chunk's 2 nearest
same-modality neighbors by position (not time) put the actual "kid with a balloon...
never ever ever ever going to let go" text into context for the first time across
every attempt so far. The resulting answer explicitly names the balloon analogy and
describes the "hold on forever" mindset, matching the expected answer's actual
content -- double-checked against the raw answer text, exactly like the rerank
result was, since one misgrade this round was reason enough to distrust every
"correct" until read. Re-running the full 6-video suite (the strict bar going in:
keep only if the podcast improves *and* nothing else regresses) showed the podcast
at a clean 4.0/4 and every other video unchanged or improved -- zero regressions,
so it's the default now, not gated behind a flag the way rerank is.

**Eval-integrity note: a judge misgrade, caught by spot-checking, not by design.**
The raw run scored the balloon question "correct" -- but reading the actual answer
("the details of that analogy are not included... I cannot provide the exact
analogy") against the actual expected answer (which describes the balloon/wrist
details explicitly) shows the judge mis-graded an "I don't know" as if it matched an
"unknown" expected answer, when the expected answer isn't "unknown" at all. Manually
corrected to "wrong" in `evals/results.json`, with the original judge reasoning kept
in the record for transparency. This is stronger than the previously-documented
judge noise-at-the-margin (a correct/partial disagreement on an otherwise-right
answer) -- this was a clear-cut error on a question where the answer's actual
content plainly didn't match. Two other score changes this round (rice's MSG
question, pasta's ingredients question) were checked the same way and found to be
legitimate, minor partial-credit calls, not misgrades.

Both blind-spot questions (fast break speed, specific dribble move) were answered
honestly by every system across every round -- Part A's blind-spot prompt logic has
been unaffected by every retrieval change since. Both systems still miss the pasta
bacon question for the unrelated, already-documented ingestion reason (see
Limitations).

**A follow-up fix: short videos were retrieving nearly the whole corpus.** TED (66
items) was pulling 45-63 items per question -- 68-95% of the video, no longer
selective, just a slower version of the naive dump. Root cause: adaptive top-k
already shrinks the *seed* selection for small corpora, but the two expansion steps
(adjacent-context, cross-modal) are proportional, not absolute -- there's less room
on a small video for them to avoid re-covering most of it.

**Fix: a ~35% relative cap on the final retrieved set**, trimming only
expansion-only additions, never the seed. Retrieved-item counts -- deterministic,
unaffected by LLM/judge noise -- are the clean signal here:

```
                TED (66 items)          podcast (911 items)
before cap    45-63 items (68-95%)     160-180 items (18-20%)
after cap     23 items (35%, capped)   159-180 items (17-20%, unchanged)
```

**Podcast is provably unaffected, not just observed to be similar.** The cap
function's first check is `if len(expanded) <= cap: return expanded` -- for podcast,
cap=319, and its expanded set has never exceeded 180 in any measurement. The
function returns the untouched set every time, for any embedding outcome; there is
no code path where podcast's retrieved content can differ from the unmodified
version.

**The eval score across 3 post-fix runs was noisier than any prior fix here
(podcast: 3.5/4, 3.5/4, 3.0/4, vs. a single 4.0/4 pre-fix measurement) -- worth
stating plainly rather than smoothing over.** Every dip traced back to the same two
pre-existing weak spots, verified by reading the actual answer text each time: the
balloon-analogy question's chronic judge nitpicking (present since the
adjacent-context work above), and, in one run, the near-silent pasta video's Whisper
transcription hallucinating *different* phantom "speech" ("Yeah" repeated, instead
of the usual "Oh." repeated) than every prior run of the same file -- which also
depressed naive's score on a code path that never calls `retrieve()` at all. No dip
was ever traced to missing or wrong retrieved content. Combined with the structural
proof above, this is accepted as pre-existing LLM/judge and Whisper non-determinism,
not evidence against the fix.

**Judge calibration.** The balloon question's chronic correct/partial flipping (see
above) was eventually traced to two separate, fixable weaknesses in the judge
prompt itself, not the system under test:

1. **The original rubric graded on vague "substance match," and the schema
   generated `verdict` before `reasoning`** -- meaning the judge picked a grade
   first and only wrote a justification for it afterward, structurally
   incapable of letting reasoning inform the verdict. Fixed by rewriting the
   rubric around an explicit CENTRAL FACT (correct = contains it regardless of
   wording; partial = touches the topic but misses/distorts it; wrong =
   contradicts or fabricates) and reordering the structured-output schema so
   `reasoning` is generated first.
2. **That fix introduced a new failure mode of its own**, caught before being
   accepted: the rubric's honesty carve-out ("an answer matching an
   expected-unknown conclusion is correct") was worded loosely enough that the
   judge sometimes applied it whenever the *actual* answer declined to
   answer -- including on the pasta ingredients question, where the *expected*
   answer states a concrete fact (egg yolks, parmesan, bacon). A flat "not
   specified" from naive was graded "correct" there, exactly the
   score-inflating failure this whole effort was trying to avoid. Fixed by
   restricting the carve-out to fire only when the *expected* answer itself
   asserts absence -- never just because the actual answer punts.
3. **Verified with two full-suite runs after the restriction**: the pasta
   ingredients question graded "wrong" for both systems in both runs (stable,
   correct). The balloon question no longer swings to "wrong" or produces
   self-contradictory reasoning -- but it still moves between "correct" and
   "partial" run to run, and reading the reasoning both times shows the judge
   weighing one specific narrative detail (whether the "tying the balloon to
   your wrist" image is present) differently depending on the exact phrasing
   generated that run. That's accepted as bounded, genuine LLM-judge
   non-determinism on a subjective edge case, not a bug -- majority-vote judging
   (already noted under Future work) is the lever to fully close it, deferred
   for now since tripling judge cost isn't worth it for one edge case.

## Limitations

- **Three distinct retrieval-at-scale failure modes were found; all three are now
  addressed, though not all by the tool first tried on them.** (1) Fixed top-k
  missing rankable-but-excluded passages -- fixed by adaptive top-k. (2) Pure
  vocabulary mismatch between question and answer wording -- a wider adaptive
  candidate pool gets the passage *into* consideration. (3) A passage-relevance
  cross-encoder isn't the right tool when the passage that *announces* an answer
  scores higher than the passage that *contains* it (a narrative-structure problem,
  not a relevance-ranking one) -- fixed not by a better reranker but by
  adjacent-context (sentence-window) retrieval, which doesn't rank at all, it just
  pulls in what's next to a winning chunk regardless of relevance score (see Eval
  results for the full case study).
- **Adjacent-context expansion adds real breadth to every retrieved context --
  the relative cap bounds the worst case (small videos), but doesn't make the
  tradeoff disappear.** Expansion stacking (same-modality neighbors + cross-modal
  matches per winning chunk) is what made TED over-retrieve in the first place;
  the ~35% cap fixes that specific failure mode without addressing whether 35% (or
  `ADJACENT_NEIGHBOR_COUNT = 2`) is the *right* amount of breadth for a given
  question -- just a bound on how much it can compound. On large videos like the
  podcast (164-180 items, ~18-20% of the corpus) the cap never engages at all,
  since expansion there naturally stays well under the threshold -- meaning any
  precision-dilution risk from a large context on a *very* long or dense video
  remains untested (this eval set's one long video tops out at ~20%, not close to
  the 35% ceiling).
- The ±10s temporal-alignment window and the 2-neighbor adjacent-context window are
  both still fixed regardless of corpus size or video length, unlike the candidate
  pool.
- **LLM-as-judge can clear-cut misgrade, not just disagree at the margin.** One eval
  run scored an evasive "I don't know" answer as "correct" against an expected answer
  that explicitly wasn't "unknown" -- caught only by manually reading the answer text
  against the expected text, not by the automated score. Manually corrected in
  `evals/results.json`; the harness has no automated safeguard against this yet, so
  any single eval run's raw numbers should be spot-checked, not trusted blindly.
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

- **Adaptive adjacent-context window.** `ADJACENT_NEIGHBOR_COUNT = 2` is currently a
  flat constant regardless of corpus size, same as the ±10s cross-modal window --
  a longer, denser video might want a different value than a short clip, but there's
  no rank-data-grounded evidence yet for what that curve should look like (one long
  video is one data point, not a curve -- see Limitations).
- **Adjustable answer depth.** Right now every answer is whatever length the LLM
  defaults to. Letting the user pick concise vs. detailed (e.g. a toggle or a
  system-prompt parameter) would help match the answer to the question -- a quick
  factual lookup and "walk me through the whole segment" shouldn't get the same
  amount of prose.
- **A "go deeper" follow-up action.** After a concise answer, offer a one-click way
  to expand it -- e.g. re-retrieve with a larger k/window scoped to the same
  timestamp range and re-answer with more detail/context, rather than making the
  user rephrase the question to get more.
- **Majority-vote judging.** Call the judge N times (e.g. 3) per answer and take
  the majority verdict, to close the residual bounded correct/partial variance on
  subjective edge cases like the balloon question (see Eval results, "Judge
  calibration") that a better rubric alone narrows but doesn't fully eliminate.
  Deferred for now -- tripling judge cost isn't worth it for one edge case.

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
local Hugging Face cache; subsequent runs reuse it. Rerank is off by default -- set
`SKIM_ENABLE_RERANK=true` in `.env` to turn it on (see `.env.example`), which
downloads the `ms-marco-MiniLM-L-6-v2` cross-encoder (~90MB) to the same cache on
first use.

To run the eval suite (needs your own video files under `evals/videos/` matching
`evals/dataset.json`; videos are gitignored and not included in this repo):

```bash
python -m evals.run_evals
```
