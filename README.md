# Skim

Chat with a video's spoken *and* visual content. Upload an `.mp4`, get a timestamped
transcript plus descriptions of key visual frames, and ask questions in a chat box —
answers cite the `[mm:ss]` timestamps they're based on, whether that's something said
or something shown.

**Scope decision:** built for talking-style videos — tutorials, talks, interviews,
lectures — where the meaning lives mostly in audio plus a few key visuals. Not built
for fast-motion or sports footage.

This is **Palier 2** of a larger plan (see `videolens-spec.md`): audio + visual
understanding. Palier 1 was audio-only. Later paliers add real retrieval (embeddings +
semantic search instead of dumping the full context) and evals. Each palier is meant
to be a complete, working project on its own — this one is it for now.

## How it works

1. **ffmpeg** extracts mono 16kHz audio from the uploaded video (`ingest/extract.py`).
2. **faster-whisper** (local, CPU, `small` model) transcribes it into timestamped
   segments (`ingest/transcribe.py`).
3. **ffmpeg scene detection** (`select='gt(scene,0.4)'`) captures a frame only when
   the image changes significantly, hard-capped at ~25 frames sampled uniformly
   across all detected changes if there are more. The video's opening frame is
   always included too, since the scene filter only fires on transitions and would
   otherwise miss it. If fewer than 3 scene changes are detected (e.g. a static
   talking-head video), uniformly time-spaced frames fill in so the visual index is
   never empty (`ingest/frames.py`).
4. **GPT-4o** describes all sampled frames in a single batched vision call —
   transcribing any on-screen text/numbers verbatim — and returns one description
   per frame via structured output (`ingest/describe.py`).
5. The transcript segments and frame descriptions are merged into one context, sorted
   chronologically and tagged `(audio)` / `(visual)`, and placed in an LLM's system
   prompt. Each chat question is answered against that fused context, citing
   `[mm:ss]` timestamps and fusing both sources when a question needs both
   (`qa/answer.py`).
6. Streamlit (`app.py`) wires this into an upload → transcript + frames → chat UI.

## Design decisions

- **faster-whisper over the OpenAI Whisper API.** Runs locally on CPU (built/tested
  on an Apple M4 Pro), so there's no per-minute transcription cost and no dependency
  on an API key for the transcription step — only the vision/chat steps need
  `OPENAI_API_KEY`.
- **Scene-detection sampling, not 1 frame/sec.** Bounds cost/latency regardless of
  video length — a hard cap on frames, not a hard cap on video duration. Frames
  beyond the cap are sampled uniformly across all detected scenes rather than just
  taking the first N, so coverage stays spread across the whole video.
- **One batched vision call, not one per frame.** Describing 25 frames as 25
  separate GPT-4o requests would pay the fixed per-request overhead (repeated system
  prompt tokens, latency) 25 times over. Sending all frames in a single request with
  structured JSON output is both cheaper and simpler to parse reliably.
- **Dump the whole fused context, no retrieval — for now.** Palier 2 is still the
  simplest thing that works: transcript + frame descriptions both fit in context for
  reasonably-sized videos. Retrieval (embed + search relevant chunks instead of
  dumping everything) is Palier 3.
- **Audio-first, visual as a fusion source.** The transcript is the map of the video;
  frame descriptions are pulled in as a second timestamped source rather than
  replacing audio, since spoken content still carries most of the meaning in
  talking-style videos.
- **Honest about stills, not motion.** Frame descriptions come from sampled still
  images, not continuous video, so the system prompt tells the model to say so when
  a question depends on motion/speed rather than guess from a snapshot. (Dedicated
  blind-spot *detection* logic is Palier 4 — this is just honest prompt wording for
  now.)

## Limitations

- No retrieval yet — the whole fused context goes into every question, which won't
  scale to very long videos (context window limits, cost, latency).
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
