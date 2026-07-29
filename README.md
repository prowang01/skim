# Skim

Chat with a video's spoken *and* visual content. Upload an `.mp4`, get a timestamped
transcript plus descriptions of key visual frames, and ask questions in a chat box —
each question retrieves the relevant moments (not the whole video) and answers cite
the `[mm:ss]` timestamps they're based on, whether that's something said or shown.

**Scope decision:** built for talking-style videos — tutorials, talks, interviews,
lectures — where the meaning lives mostly in audio plus a few key visuals. Not built
for fast-motion or sports footage.

This is **Palier 3** of a larger plan (see `videolens-spec.md`): retrieval + smarter
fusion. Palier 1 was audio-only; Palier 2 added visual descriptions but still dumped
the whole transcript + all frames into every question. Palier 4 (blind-spot detection
logic + evals) is next. Each palier is meant to be a complete, working project on its
own — this one is it for now.

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
   chunk alone wouldn't have matched the question's wording — this is what lets
   "look at the screen" (audio) get paired with the frame that actually shows what's
   on screen, and lets the model reason about the two together
   (`index/retrieve.py`).
7. The retrieved, temporally-aligned items are sorted chronologically and placed in
   an LLM's system prompt, which explicitly instructs it to treat close-in-time
   audio/visual lines as the same moment and infer what's being *done* — not just
   describe the frame in isolation. It also cites `[mm:ss]` timestamps and is honest
   when the retrieved excerpt (not the full video) doesn't contain the answer
   (`qa/answer.py`).
8. Streamlit (`app.py`) wires this into an upload → transcript + frames → chat UI,
   with an expander showing exactly what was retrieved for each question.

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
- **Retrieval, not dump.** This is the core change from Palier 2: only the top-k
  chunks relevant to a specific question go into the prompt, not the entire
  transcript and all frame descriptions. This is what actually stops the design
  from breaking down on longer videos, and it's the difference between "a script"
  and a RAG system.
- **Retrieve-then-expand for fusion, not retrieve-then-dump.** Ranking audio and
  visual chunks purely by semantic similarity to the question tends to miss frame
  descriptions that don't share the question's vocabulary (e.g. "shows 4 egg yolks
  on screen" vs. a question like "what are the exact quantities") even when they're
  exactly the moment being asked about. Expanding each retrieved chunk with
  same-time-window chunks from the other modality fixes this without expanding the
  whole context — it's targeted, not a second dump.
- **Honest about stills, not motion, and about being an excerpt.** Frame
  descriptions come from sampled stills, not continuous video, so the prompt says so
  for motion/speed questions. And because retrieval means the model no longer sees
  the whole video, the prompt also tells it the retrieved excerpt might genuinely be
  incomplete — it shouldn't assume it saw everything. (Dedicated blind-spot
  *detection* logic and evals to measure this against the Palier-1 baseline are
  Palier 4.)

## Limitations

- Retrieval is still simple: fixed top-k (6) and a fixed ±10s alignment window, not
  adapted to video length or content density.
- Visual understanding is frame-sampling based: up to ~25 stills per video, so
  anything that happens between sampled frames (or during fast motion) may be missed
  entirely.
- Local transcription runs on CPU; a long video will take a while on first upload.
  Test with a short (2-3 min) clip first.
- The `small` faster-whisper model trades some accuracy for speed — expect
  occasional mis-transcriptions, especially with accents, cross-talk, or background
  noise. Whisper's language auto-detection can also occasionally mis-fire on short
  or unusual-sounding audio.
- Frame descriptions are only as good as GPT-4o's read of the image — small or
  low-contrast on-screen text may still be misread.
- No evals yet — Palier 3's retrieval/fusion quality has been spot-checked manually,
  not measured against a hand-built test set. That comparison (vs. the Palier-1
  dump-everything baseline) is Palier 4.

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
