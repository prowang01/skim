"""Speech-to-text transcription via faster-whisper (local, CPU)."""

from dataclasses import dataclass

from faster_whisper import WhisperModel

DEFAULT_MODEL_SIZE = "small"

_model_cache: dict[str, WhisperModel] = {}


@dataclass
class Segment:
    start: float
    end: float
    text: str


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
    ISO 639-1 code (e.g. "en", "fr"); None lets Whisper auto-detect, which
    occasionally misfires on short or unusual-sounding audio."""
    model = _get_model(model_size)
    # beam_size=5 is faster-whisper's own default, not separately tuned here.
    segments, _info = model.transcribe(audio_path, beam_size=5, language=language)
    return [Segment(start=s.start, end=s.end, text=s.text.strip()) for s in segments]
