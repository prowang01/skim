"""Scene-detected frame extraction via ffmpeg."""

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

SCENE_THRESHOLD = 0.4
MAX_FRAMES = 25
MIN_SCENE_FRAMES_BEFORE_FALLBACK = 3


@dataclass
class Frame:
    timestamp: float
    path: str


def _probe_duration(video_path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def _extract_scene_frames(video_path: str, out_dir: Path) -> list[Frame]:
    pattern = out_dir / "scene_%04d.jpg"
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", f"select='gt(scene,{SCENE_THRESHOLD})',showinfo",
            "-vsync", "vfr", "-q:v", "2",
            str(pattern),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg scene detection failed:\n{result.stderr}")

    # showinfo logs one "pts_time:" line per frame that passed the select
    # filter, in order, on stderr -- that's how we recover timestamps for
    # ffmpeg's auto-numbered output files.
    timestamps = [float(m.group(1)) for m in re.finditer(r"pts_time:([\d.]+)", result.stderr)]
    frame_files = sorted(out_dir.glob("scene_*.jpg"))

    n = min(len(timestamps), len(frame_files))
    return [Frame(timestamp=timestamps[i], path=str(frame_files[i])) for i in range(n)]


def _extract_uniform_frames(
    video_path: str, out_dir: Path, count: int, duration: float
) -> list[Frame]:
    if count <= 0 or duration <= 0:
        return []

    frames = []
    for i in range(count):
        t = duration * (i + 0.5) / count
        out_file = out_dir / f"uniform_{i:04d}.jpg"
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-ss", str(t), "-i", video_path,
                "-frames:v", "1", "-q:v", "2",
                str(out_file),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and out_file.exists():
            frames.append(Frame(timestamp=t, path=str(out_file)))
    return frames


def _extract_first_frame(video_path: str, out_dir: Path) -> Frame | None:
    out_file = out_dir / "opening.jpg"
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-frames:v", "1", "-q:v", "2", str(out_file)],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and out_file.exists():
        return Frame(timestamp=0.0, path=str(out_file))
    return None


def _sample_uniform_subset(frames: list[Frame], cap: int) -> list[Frame]:
    """Spread a hard cap across the whole list instead of truncating to the
    first `cap` entries, so coverage stays spread across the video."""
    if len(frames) <= cap:
        return frames

    indices = sorted({round(i * (len(frames) - 1) / (cap - 1)) for i in range(cap)})
    return [frames[i] for i in indices]


def extract_frames(video_path: str, out_dir: str) -> list[Frame]:
    """Extract up to MAX_FRAMES timestamped frames from a video.

    Prefers scene-change frames (only capture when the image changes
    significantly) over fixed-interval sampling, so cost is bounded on long
    videos without missing real cuts. Falls back to uniform time sampling
    if too few scene changes are detected (e.g. a static talking-head
    video), so the visual index is never empty.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    frames = _extract_scene_frames(video_path, out_path)

    # The scene filter only fires on transitions, so the video's opening
    # visual (before the first detected cut) is otherwise never captured.
    first_timestamp = frames[0].timestamp if frames else float("inf")
    if first_timestamp > 1.0:
        opening = _extract_first_frame(video_path, out_path)
        if opening is not None:
            frames.insert(0, opening)

    if len(frames) < MIN_SCENE_FRAMES_BEFORE_FALLBACK:
        duration = _probe_duration(video_path)
        frames += _extract_uniform_frames(video_path, out_path, MAX_FRAMES - len(frames), duration)
        frames.sort(key=lambda f: f.timestamp)

    return _sample_uniform_subset(frames, MAX_FRAMES)
