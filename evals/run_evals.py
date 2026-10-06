"""Eval harness: run the retrieval-based system and the naive
(dump-everything) baseline against evals/dataset.json, grade both with an
LLM judge, and print a comparison.

Transcripts are cached by video content hash and STT provider. Frame
descriptions are shared across providers. Retrieval, answers, and judging run fresh
so you can iterate on top-k/cap/adjacent-context/the judge rubric without
re-transcribing.

Usage:
  python -m evals.run_evals              # full 7-video suite -- run before committing
  python -m evals.run_evals --fast       # FAST_VIDEO_IDS only -- for iteration
  python -m evals.run_evals --video ted  # one-video smoke test
  python -m evals.run_evals --no-cache   # force fresh ingestion (e.g. after changing
                                          # transcribe.py/frames.py/describe.py)
  python -m evals.run_evals --clear-cache  # wipe the ingestion cache and exit
"""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from ingest.extract import extract_audio
from ingest.transcribe import (
    transcribe, Segment, get_stt_provider, DEFAULT_MODEL_SIZE, ELEVENLABS_MODEL,
)
from ingest.frames import extract_frames
from ingest.describe import describe_frames, FrameDescription
from index.build_index import build_index
from index.retrieve import retrieve
from qa.answer import answer_question
from evals.naive_baseline import answer_naive

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = Path(__file__).parent / "dataset.json"

# Quick-iteration subset for --fast: the podcast's 32min transcription is the
# bottleneck in the full suite (~10min end to end). These are short clips,
# covering English + French (gelee_groseille) without touching the podcast.
FAST_VIDEO_IDS = ["rice", "ted", "gelee_groseille"]

# Bump this if the cached ingestion shape ever changes (e.g. adding per-segment
# confidence scores) so old cache entries get ignored instead of misread.
CACHE_VERSION = 1
CACHE_DIR = Path(__file__).parent / ".cache"

JUDGE_MODEL = "gpt-4o-mini"
SCORE_MAP = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}

JUDGE_SYSTEM_PROMPT = """You are grading a video-QA system's answer against a \
hand-written expected answer.

First identify the CENTRAL FACT of the expected answer -- the one thing the \
question is actually asking for. Then check whether the actual answer contains \
that central fact, regardless of wording. A differently-worded answer that \
states the same central fact is just as correct as one that echoes the \
expected phrasing.

Grade using this rubric exactly:

- "correct": the actual answer contains the central fact of the expected \
answer, even if it omits secondary details the expected answer mentions, uses \
different wording, or adds extra (correct) context.
- "partial": the actual answer touches the right topic or moment but misses or \
distorts the central fact itself -- e.g. it's vague exactly where the central \
fact needs to be specific, or it gets a secondary detail right while getting \
the central fact wrong.
- "wrong": the actual answer contradicts the central fact, is off-topic, or \
confidently fabricates information the expected answer says isn't available.

For questions where the expected answer's central fact is that something is \
NOT known/shown/said, an actual answer that reaches that same conclusion in \
its own words is "correct" -- it does not need to match the expected wording, \
only the conclusion.

This "not known" exception applies ONLY when the EXPECTED answer itself \
asserts that the information is absent/unavailable/not shown. It does NOT \
apply just because the ACTUAL answer declines to answer or claims not to have \
the information. If the expected answer states a concrete fact (a name, a \
number, an ingredient, an analogy, etc.), an actual answer that says "not \
specified" / "I don't know" / "not provided in this excerpt" is WRONG, full \
stop -- declining to state a knowable fact is a failure to answer, not \
honesty, and must never be graded "correct" or "partial" just for admitting \
uncertainty.

Be consistent: apply the same standard to every answer. This is not about \
being lenient or strict -- it's about whether the central fact is actually \
present. State your reasoning (including what you judged the central fact to \
be) before giving your verdict.
"""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "verdict": {"type": "string", "enum": ["correct", "partial", "wrong"]},
    },
    "required": ["reasoning", "verdict"],
    "additionalProperties": False,
}


def judge(question: str, expected: str, actual: str) -> dict:
    """Grade an answer against the expected answer; returns {verdict, reasoning}."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question: {question}\n\nExpected answer: {expected}\n\nActual answer: {actual}",
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "verdict", "strict": True, "schema": JUDGE_SCHEMA},
        },
    )
    return json.loads(response.choices[0].message.content)


def _file_hash(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_path(video_path: str) -> Path:
    return CACHE_DIR / get_stt_provider() / f"{Path(video_path).stem}_{_file_hash(video_path)[:16]}.json"


def _cached_frames(cache_path: Path) -> list[FrameDescription] | None:
    # Only visuals may cross provider boundaries. Legacy unlabelled caches
    # can supply visuals, but their transcripts are never read here.
    for directory in ("visual", "", "whisper", "elevenlabs"):
        path = CACHE_DIR / directory / cache_path.name
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("cache_version") == CACHE_VERSION:
                return [FrameDescription(**f) for f in cached["frame_descriptions"]]
    return None


def ingest_video(video_path: str, use_cache: bool = True):
    """Run the full ingestion pipeline (audio, transcript, frames,
    descriptions) and build the semantic index for one video. Ingestion is
    cached by video content hash and provider; visuals are shared between
    providers to isolate the STT change. Embedding/indexing still happens
    fresh every call regardless of cache hits."""
    provider = get_stt_provider()
    cache_path = _cache_path(video_path)

    if use_cache and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("cache_version") == CACHE_VERSION and cached.get("stt_provider") == provider:
            print(f"  (ingestion cache hit: {cache_path.name})", flush=True)
            segments = [Segment(**s) for s in cached["segments"]]
            descriptions = [FrameDescription(**f) for f in cached["frame_descriptions"]]
            return build_index(segments, descriptions)

    descriptions = _cached_frames(cache_path) if use_cache else None
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        audio_path = tmp_path / "audio.wav"
        frames_dir = tmp_path / "frames"

        extract_audio(video_path, str(audio_path))
        segments = transcribe(str(audio_path))
        if descriptions is None:
            frames = extract_frames(video_path, str(frames_dir))
            descriptions = describe_frames(frames)

    visual_path = CACHE_DIR / "visual" / cache_path.name
    visual_path.parent.mkdir(parents=True, exist_ok=True)
    visual_path.write_text(json.dumps({
        "cache_version": CACHE_VERSION,
        "frame_descriptions": [
            {"timestamp": f.timestamp, "description": f.description} for f in descriptions
        ],
    }, indent=2), encoding="utf-8")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "cache_version": CACHE_VERSION,
                "stt_provider": provider,
                "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in segments],
                "frame_descriptions": [
                    {"timestamp": f.timestamp, "description": f.description} for f in descriptions
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return build_index(segments, descriptions)


def _score(rows: list[dict]) -> tuple[float, int]:
    return sum(SCORE_MAP[r["verdict"]] for r in rows), len(rows)


def print_breakdown(results: list[dict], group_key: str, title: str) -> None:
    systems = ["retrieval", "naive"]
    groups = sorted({r[group_key] for r in results})

    print()
    print("=" * 50)
    print(f"{title:<20}{'retrieval':>15}{'naive':>15}")
    for group in groups:
        row = f"{group:<20}"
        for system in systems:
            rows = [r for r in results if r[group_key] == group and r["system"] == system]
            total, n = _score(rows)
            row += f"{f'{total:.1f}/{n}':>15}"
        print(row)
    print("-" * 50)
    row = f"{'TOTAL':<20}"
    for system in systems:
        rows = [r for r in results if r["system"] == system]
        total, n = _score(rows)
        row += f"{f'{total:.1f}/{n}':>15}"
    print(row)
    print("=" * 50)


def print_summary(results: list[dict]) -> None:
    print_breakdown(results, "video", "video")
    print_breakdown(results, "category", "category")


def run(video_ids: list[str] | None = None, use_cache: bool = True) -> list[dict]:
    """Run every question for the given video ids (or all videos, if None)
    through both systems, grade both, write the results file, and print the
    score breakdown."""
    provider = get_stt_provider()
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    videos = dataset["videos"]
    if video_ids is not None:
        unknown = set(video_ids) - {v["id"] for v in videos}
        if unknown or not video_ids:
            raise ValueError(f"Invalid video selection: {video_ids}")
        videos = [v for v in videos if v["id"] in video_ids]
        print(f"=== Selected videos: {[v['id'] for v in videos]} ===", flush=True)
    print(f"=== STT provider: {provider} ===", flush=True)

    results = []

    for video in videos:
        video_path = str(PROJECT_ROOT / video["path"])
        print(f"=== Ingesting {video['id']} ===", flush=True)
        index = ingest_video(video_path, use_cache=use_cache)
        print(f"  {len(index.items)} indexed items", flush=True)

        for q in video["questions"]:
            question, expected, category = q["question"], q["expected"], q["category"]
            print(f"  Q ({category}): {question}", flush=True)

            retrieved = retrieve(index, question)
            retrieval_answer = answer_question(retrieved, question)
            naive_answer = answer_naive(index, question)

            retrieval_verdict = judge(question, expected, retrieval_answer)
            naive_verdict = judge(question, expected, naive_answer)

            for system, answer, verdict in [
                ("retrieval", retrieval_answer, retrieval_verdict),
                ("naive", naive_answer, naive_verdict),
            ]:
                print(f"    {system:<10} {verdict['verdict']}", flush=True)
                results.append(
                    {
                        "stt_provider": provider,
                        "stt_model": DEFAULT_MODEL_SIZE if provider == "whisper" else ELEVENLABS_MODEL,
                        "video": video["id"],
                        "question": question,
                        "expected": expected,
                        "category": category,
                        "system": system,
                        "answer": answer,
                        "verdict": verdict["verdict"],
                        "reasoning": verdict["reasoning"],
                    }
                )

    suffix = ""
    if video_ids is not None:
        suffix = "_fast" if video_ids == FAST_VIDEO_IDS else "_" + "_".join(v["id"] for v in videos)
    results_path = Path(__file__).parent / f"results_{provider}{suffix}.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Results saved to {results_path}", flush=True)
    print_summary(results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--fast",
        action="store_true",
        help=f"Run only {FAST_VIDEO_IDS} (writes results_<provider>_fast.json)",
    )
    selection.add_argument(
        "--video",
        choices=[v["id"] for v in json.loads(DATASET_PATH.read_text(encoding="utf-8"))["videos"]],
        help="Run one video (writes results_<provider>_<video>.json)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force fresh ingestion instead of using the cache (still refreshes the cache for next time)",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Delete the ingestion cache and exit, without running evals",
    )
    args = parser.parse_args()

    if args.clear_cache:
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
        print(f"Cleared {CACHE_DIR}")
    else:
        video_ids = [args.video] if args.video else FAST_VIDEO_IDS if args.fast else None
        run(video_ids=video_ids, use_cache=not args.no_cache)
