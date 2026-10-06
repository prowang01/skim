"""Speech-to-text via local faster-whisper or ElevenLabs Scribe."""

import os
from dataclasses import dataclass

from faster_whisper import WhisperModel

DEFAULT_MODEL_SIZE = "small"
ELEVENLABS_MODEL = "scribe_v2"

_model_cache: dict[str, WhisperModel] = {}


@dataclass
class Segment:
    start: float
    end: float
    text: str


def get_stt_provider() -> str:
    provider = os.environ.get("SKIM_STT_PROVIDER", "whisper").strip().lower()
    if provider not in ("whisper", "elevenlabs"):
        raise ValueError("SKIM_STT_PROVIDER must be 'whisper' or 'elevenlabs'")
    return provider


def _transcribe_elevenlabs(audio_path: str, language: str | None) -> list[Segment]:
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("Set ELEVENLABS_API_KEY in .env to use ElevenLabs STT")

    from elevenlabs.client import ElevenLabs

    # Long recordings can exceed the SDK's default HTTP timeout.
    client = ElevenLabs(api_key=api_key, timeout=1800)
    with open(audio_path, "rb") as audio:
        transcript = client.speech_to_text.convert(
            file=audio,
            model_id=ELEVENLABS_MODEL,
            language_code=language,
            timestamps_granularity="word",
            tag_audio_events=False,
            diarize=False,
        )

    # Preserve API spacing and punctuation, but embed speech-sized chunks,
    # not individual words. Bound long unpunctuated passages to 10 seconds.
    segments = []
    parts: list[str] = []
    start = end = 0.0
    for word in transcript.words:
        if word.type == "spacing":
            if parts:
                parts.append(word.text)
            continue
        if word.type != "word":
            continue
        if word.start is None or word.end is None:
            raise RuntimeError("ElevenLabs returned speech without word timestamps")
        if parts and word.start - end >= 1.0:
            segments.append(Segment(start, end, "".join(parts).strip()))
            parts = []
        if not parts:
            start = word.start
        parts.append(word.text)
        end = word.end
        if word.text.rstrip().endswith((".", "?", "!", "…")) or end - start >= 10.0:
            segments.append(Segment(start, end, "".join(parts).strip()))
            parts = []
    if parts:
        segments.append(Segment(start, end, "".join(parts).strip()))
    if transcript.text.strip() and not segments:
        raise RuntimeError("ElevenLabs returned text without timestamped speech")
    return segments


def _get_model(model_size: str) -> WhisperModel:
    # CTranslate2 has no Metal backend, so this runs on CPU; int8 keeps it fast.
    if model_size not in _model_cache:
        _model_cache[model_size] = WhisperModel(
            model_size, device="cpu", compute_type="int8"
        )
    return _model_cache[model_size]


def transcribe(
    audio_path: str, model_size: str = DEFAULT_MODEL_SIZE, language: str | None = None
) -> list[Segment]:
    """Transcribe an audio file into timestamped segments. language is an
    ISO 639-1 code (e.g. "en", "fr"); None lets the provider auto-detect, which
    occasionally misfires on short or unusual-sounding audio."""
    if get_stt_provider() == "elevenlabs":
        return _transcribe_elevenlabs(audio_path, language)

    model = _get_model(model_size)
    # beam_size=5 is faster-whisper's own default, not separately tuned here.
    segments, _info = model.transcribe(audio_path, beam_size=5, language=language)
    return [Segment(start=s.start, end=s.end, text=s.text.strip()) for s in segments]
