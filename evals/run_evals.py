"""Eval harness: run the retrieval-based system and the naive
(dump-everything) baseline against evals/dataset.json, grade both with an
LLM judge, and print a comparison.

Usage:
  python -m evals.run_evals            # full 6-video suite -- run before committing
  python -m evals.run_evals --fast     # FAST_VIDEO_IDS only (~30s) -- for iteration
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from ingest.extract import extract_audio
from ingest.transcribe import transcribe
from ingest.frames import extract_frames
from ingest.describe import describe_frames
from index.build_index import build_index
from index.retrieve import retrieve
from qa.answer import answer_question
from evals.naive_baseline import answer_naive

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = Path(__file__).parent / "dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.json"
FAST_RESULTS_PATH = Path(__file__).parent / "results_fast.json"

# Quick-iteration subset for --fast: the podcast's 32min transcription is the
# bottleneck in the full suite (~10min end to end). These two are the smallest
# short clips, covering 2 categories each without touching the podcast.
FAST_VIDEO_IDS = ["rice", "ted"]

JUDGE_MODEL = "gpt-4o-mini"
SCORE_MAP = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}

JUDGE_SYSTEM_PROMPT = """You are grading a video-QA system's answer against a \
hand-written expected answer. Grade the ACTUAL answer as one of:

- "correct": the actual answer's substance matches the expected answer -- same \
facts, same conclusion. Minor wording differences or extra detail don't count \
against it.
- "partial": the actual answer gets part of it right but is incomplete, hedges \
when it shouldn't, or includes a real error alongside correct content.
- "wrong": the actual answer contradicts the expected answer, confidently \
fabricates information the expected answer says isn't available, or misses the \
point entirely.

For questions where the expected answer says something is NOT known/shown/said, \
an actual answer that reaches that same conclusion in its own words is "correct" \
-- it does not need to match the expected wording, only the conclusion.
"""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["correct", "partial", "wrong"]},
        "reasoning": {"type": "string"},
    },
    "required": ["verdict", "reasoning"],
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


def ingest_video(video_path: str):
    """Run the full ingestion pipeline (audio, transcript, frames,
    descriptions) and build the semantic index for one video."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        audio_path = tmp_path / "audio.wav"
        frames_dir = tmp_path / "frames"

        extract_audio(video_path, str(audio_path))
        segments = transcribe(str(audio_path))
        frames = extract_frames(video_path, str(frames_dir))
        descriptions = describe_frames(frames)

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


def run(video_ids: list[str] | None = None) -> list[dict]:
    """Run every question for the given video ids (or all videos, if None)
    through both systems, grade both, write the results file, and print the
    score breakdown."""
    dataset = json.loads(DATASET_PATH.read_text())
    videos = dataset["videos"]
    if video_ids is not None:
        videos = [v for v in videos if v["id"] in video_ids]
        print(f"=== FAST mode: {[v['id'] for v in videos]} only -- run without --fast for the full suite before committing ===", flush=True)

    results = []

    for video in videos:
        video_path = str(PROJECT_ROOT / video["path"])
        print(f"=== Ingesting {video['id']} ===", flush=True)
        index = ingest_video(video_path)
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

    results_path = FAST_RESULTS_PATH if video_ids is not None else RESULTS_PATH
    results_path.write_text(json.dumps(results, indent=2))
    print_summary(results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fast",
        action="store_true",
        help=f"Run only {FAST_VIDEO_IDS} for quick iteration (writes evals/results_fast.json instead of evals/results.json)",
    )
    args = parser.parse_args()
    run(video_ids=FAST_VIDEO_IDS if args.fast else None)
