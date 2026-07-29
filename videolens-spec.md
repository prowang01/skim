# VideoLens — Multimodal Video Understanding (build spec)

> A system that "understands" a video by building a **timestamped multimodal index** (spoken audio + key visual frames), retrieves the relevant pieces on demand to answer questions, fuses audio + visual, and is honest about its blind spots.

**Scope decision (state this in the README):** optimized for talking-style videos — tutorials, talks, interviews, lectures — where meaning lives mostly in audio plus a few key visuals. Not designed for fast-motion / sports; that limitation is acknowledged by design.

---

## Architecture (one-liner per layer)

1. **Ingestion** — mp4 in → ffmpeg splits audio + video → Whisper transcribes audio into timestamped segments.
2. **Index** — transcript segments embedded for semantic search; scene-detected frames described by a multimodal model, each description timestamped + indexed.
3. **Retrieval + answer** — a question retrieves the relevant transcript chunks and/or frame descriptions, fuses them into a timestamped context, LLM answers with timestamp sources.
4. **Blind-spot awareness** — for motion/temporal questions, the system flags that it reasons on stills, not continuous motion.

Cross-cutting: **evals** — a small hand-built test set to measure answer quality and justify design choices.

---

## Stack

- **ffmpeg** — audio extraction + frame extraction via scene detection
- **Whisper** — transcription (OpenAI API `whisper-1`, or `faster-whisper` locally to avoid cost)
- **Multimodal LLM** — GPT-4o (or Claude) for frame description + final answers
- **Embeddings** — OpenAI `text-embedding-3-small` (cheap, good enough)
- **Vector store** — start with in-memory (numpy cosine similarity or FAISS); no DB needed for a demo
- **Frontend** — Streamlit for speed (single file), OR FastAPI + React if you want to show full-stack like RoleRadar
- **Python** — the whole thing

---

## Key engineering decisions (put these in the README — they ARE the project)

- **Scene-detection sampling, not 1 frame/sec.** Capture a frame only when the image changes significantly (`ffmpeg` `select='gt(scene,0.4)'` or PySceneDetect). Hard cap at ~25 frames regardless of duration; if more are detected, sample uniformly within them. → bounds cost/latency, solves the "12-minute video" problem.
- **Audio-first indexing.** The transcript is the map of the video; visual frames are secondary and pulled in only when a question needs them.
- **Retrieval, not dump.** Never send the whole transcript + all frames to the model. Retrieve the relevant pieces per question. → the core "AI-in-prod" signal.
- **Timestamped fusion.** Answers cite timestamps (e.g. "at 2:38 the on-screen recipe shows..."), so the user can verify.
- **Honest blind spots.** The system says when it can't know (fast motion, off-screen info).

---

## Suggested folder structure

```
videolens/
  ingest/
    extract.py        # ffmpeg: audio + scene-detected frames
    transcribe.py     # Whisper -> timestamped segments
    describe.py       # multimodal model -> frame descriptions (timestamped)
  index/
    build_index.py    # embed transcript chunks + frame descriptions
    retrieve.py       # semantic search over the index
  qa/
    answer.py         # fuse retrieved context -> LLM answer w/ timestamps + blind-spot logic
  evals/
    dataset.json      # {video, question, expected} cases
    run_evals.py      # score answers, compare vs naive baseline
  app.py              # Streamlit UI (upload -> progress -> chat)
  README.md
  requirements.txt
  .env.example        # OPENAI_API_KEY=
```

---

## Build in PALIERS — each one is a complete, pushable project

### Palier 1 — Audio MVP (your safety net)
Upload mp4 → ffmpeg extracts audio → Whisper transcript → paste full transcript into LLM → chat about it.
**Done = pushable.** You already have a working "chat with a video's speech" tool.

### Palier 2 — Go multimodal
Add scene-detected frame extraction + multimodal descriptions. Fuse transcript + frame descriptions (both timestamped) into the context. Now it reasons about what's *shown*, not just said.

### Palier 3 — Real system (retrieval)
Embed transcript chunks + frame descriptions into an index. On each question, retrieve the top-k relevant pieces instead of dumping everything. This is multimodal RAG over video — the part that impresses.

### Palier 4 — AI Engineer polish
- Blind-spot logic: detect motion/temporal questions, respond honestly about stills-only reasoning.
- Evals: 4-5 videos, ~10 hand-written question/expected pairs, score your system, and compare against the Palier-1 "dump everything" baseline. Report the numbers.

Climb in order. Push at each palier. Stop wherever time runs out — you always have something that works.

---

## README checklist (this carries ~50% of the impression)

- One-sentence what/who + the scope decision (talking-style videos)
- Architecture diagram (the 4 layers)
- "Design decisions" section (the bullets above — the *why*, not just the *what*)
- "How it works" walkthrough with a concrete example (e.g. "what are the exact quantities?" → audio points to screen → frame@2:38 has the data → fused answer)
- "Limitations" section (fast motion, off-screen info, frame-sampling trade-offs) — honesty is a feature
- Eval results (even small: "9/10 vs 6/10 for the naive baseline")
- Run locally + a demo GIF placeholder

---

## Concrete example (the flow to demo)

Cooking tutorial, 10 min.
- **"Does he use cream?"** → transcript hit at 00:12 → *"No — at 00:12 he says it's the real carbonara, no cream."* (no image needed)
- **"What are the exact quantities?"** → transcript says "look on screen" (02:30) but the data is in frame@02:38 → *"Per the on-screen recipe at 2:38: 4 egg yolks, 100g pecorino, 50g parmesan."* (audio + visual fused — the money shot)
- **"Does he mix fast or slow?"** → only a still at 05:12 → *"I see a snapshot at 5:12 but reason on stills, not motion, so I can't judge speed reliably."* (blind-spot honesty)

---

## Watch-outs

- Don't chase "understands any video." A focused system that nails tutorials/talks beats an ambitious one that's mediocre everywhere.
- Don't over-sample frames. The cap is a feature, not a limitation.
- Keep the vector store in-memory for the demo — no DB rabbit hole.
- If time is short, Palier 2 is already a good project. Palier 3 = very good. Palier 4 = banger.
- Whisper on long audio can be slow/costly — test on a 2-3 min clip first, scale up after.
