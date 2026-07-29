# Skim

Chat with a video's spoken content. Upload an `.mp4`, get a timestamped transcript,
and ask questions in a chat box — answers cite the `[mm:ss]` timestamps they're based on.

**Scope decision:** built for talking-style videos — tutorials, talks, interviews,
lectures — where the meaning lives mostly in what's said. Not built for fast-motion
or sports footage.

This is **Palier 1** of a larger plan (see `videolens-spec.md`): the audio-only MVP.
Later paliers add visual understanding (scene-detected frames + multimodal
descriptions), real retrieval (embeddings + semantic search instead of dumping the
full transcript), and evals. Each palier is meant to be a complete, working project
on its own — this one is it for now.

## How it works

1. **ffmpeg** extracts mono 16kHz audio from the uploaded video (`ingest/extract.py`).
2. **faster-whisper** (local, CPU, `small` model) transcribes it into timestamped
   segments (`ingest/transcribe.py`).
3. The full transcript — timestamps and all — is placed in an LLM's system prompt.
   Each chat question is answered against that transcript, with instructions to cite
   `[mm:ss]` timestamps and to say plainly when the transcript doesn't have the
   answer (`qa/answer.py`).
4. Streamlit (`app.py`) wires this into an upload → transcript → chat UI.

## Design decisions

- **faster-whisper over the OpenAI Whisper API.** Runs locally on CPU (this was
  built/tested on an Apple M4 Pro), so there's no per-minute transcription cost and
  no dependency on an API key for the transcription step — only the chat step needs
  `OPENAI_API_KEY`.
- **Dump the whole transcript, no retrieval — for now.** Palier 1 is intentionally
  the simplest thing that works: the full transcript fits in context for
  reasonably-sized videos, so it's pasted in whole. Retrieval (embed + search
  relevant chunks instead of dumping everything) is Palier 3 — the point where this
  stops being "a script" and starts being a RAG system.
- **Audio-only, and it says so.** If a question depends on something visual (on-screen
  text, what's shown, gestures), the system is instructed to say it can't see the
  video rather than guess. Visual understanding is Palier 2.

## Limitations

- No visual understanding yet — only what was said, not what was shown.
- No retrieval yet — the whole transcript goes into every question's context, which
  won't scale to very long videos (context window limits, cost, latency).
- Local transcription runs on CPU; a long video will take a while on first upload.
  Test with a short (2-3 min) clip first.
- The `small` faster-whisper model trades some accuracy for speed — expect
  occasional mis-transcriptions, especially with accents, cross-talk, or background
  noise.

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
