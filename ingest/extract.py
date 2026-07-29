"""Audio extraction from video files via ffmpeg."""

import subprocess
from pathlib import Path


def extract_audio(video_path: str, audio_path: str) -> None:
    """Extract mono 16kHz WAV audio from a video file using ffmpeg.

    16kHz mono is the sample rate/channel layout Whisper models expect,
    so we convert at extraction time instead of letting the model resample.
    """
    Path(audio_path).parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-acodec", "pcm_s16le",
            audio_path,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to extract audio:\n{result.stderr}")
