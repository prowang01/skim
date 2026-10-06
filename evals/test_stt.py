"""Offline regression checks for STT switching and experiment isolation."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx

from evals import run_evals as ev
from ingest import transcribe as stt
from ingest.describe import FrameDescription


def word(text, start, end, kind="word"):
    return SimpleNamespace(text=text, start=start, end=end, type=kind)


class TranscriptionTests(unittest.TestCase):
    def test_whisper_is_default_and_preserves_arguments(self):
        model = Mock()
        model.transcribe.return_value = ([word(" hello ", 1, 2)], None)
        with patch.dict(os.environ, {}, clear=True), patch.object(stt, "_get_model", return_value=model) as get:
            self.assertEqual(stt.transcribe("audio.wav", language="fr"), [stt.Segment(1, 2, "hello")])
        get.assert_called_once_with("small")
        model.transcribe.assert_called_once_with("audio.wav", beam_size=5, language="fr")

    def test_invalid_provider_and_missing_key_fail(self):
        with patch.dict(os.environ, {"SKIM_STT_PROVIDER": "invalid"}):
            with self.assertRaises(ValueError):
                stt.get_stt_provider()
        with patch.dict(os.environ, {"SKIM_STT_PROVIDER": "elevenlabs", "ELEVENLABS_API_KEY": ""}):
            with self.assertRaisesRegex(ValueError, "ELEVENLABS_API_KEY"):
                stt.transcribe("unused.wav")

    def test_elevenlabs_groups_words_and_preserves_timestamps(self):
        words = [word("Bonjour", 0, .4), word(" ", .4, .5, "spacing"),
                 word("monde!", .5, 1), word(" ", 1, 1, "spacing"),
                 word("Encore", 2, 2.4), word(" ", 2.4, 2.5, "spacing"),
                 word("ici", 2.5, 3), word(" ", 3, 4.5, "spacing"),
                 word("Après", 4.5, 5), word("(music)", 5, 6, "audio_event")]
        client = Mock()
        client.speech_to_text.convert.return_value = SimpleNamespace(words=words, text="Bonjour monde! Encore ici Après")
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "audio.wav"
            audio.write_bytes(b"audio")
            with patch.dict(os.environ, {"SKIM_STT_PROVIDER": "elevenlabs", "ELEVENLABS_API_KEY": "test-key"}), patch("elevenlabs.client.ElevenLabs", return_value=client) as sdk:
                segments = stt.transcribe(str(audio), language="fr")
        self.assertEqual(segments, [stt.Segment(0, 1, "Bonjour monde!"), stt.Segment(2, 3, "Encore ici"), stt.Segment(4.5, 5, "Après")])
        sdk.assert_called_once_with(api_key="test-key", timeout=1800)
        options = client.speech_to_text.convert.call_args.kwargs
        self.assertEqual(options["model_id"], "scribe_v2")
        self.assertEqual(options["language_code"], "fr")
        self.assertEqual(options["timestamps_granularity"], "word")
        self.assertFalse(options["tag_audio_events"])
        self.assertFalse(options["diarize"])
        self.assertTrue(options["file"].closed)

    def test_empty_audio_and_missing_timestamps(self):
        client = Mock()
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "audio.wav"
            audio.touch()
            with patch.dict(os.environ, {"SKIM_STT_PROVIDER": "elevenlabs", "ELEVENLABS_API_KEY": "test-key"}), patch("elevenlabs.client.ElevenLabs", return_value=client):
                client.speech_to_text.convert.return_value = SimpleNamespace(words=[], text="")
                self.assertEqual(stt.transcribe(str(audio)), [])
                client.speech_to_text.convert.return_value = SimpleNamespace(words=[word("hello", None, None)], text="hello")
                with self.assertRaisesRegex(RuntimeError, "timestamps"):
                    stt.transcribe(str(audio))
                client.speech_to_text.convert.return_value = SimpleNamespace(words=[], text="hello")
                with self.assertRaisesRegex(RuntimeError, "timestamped"):
                    stt.transcribe(str(audio))

    def test_real_sdk_request_parsing_and_long_passage_split(self):
        from elevenlabs.client import ElevenLabs

        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(200, json={
                "language_code": "en", "language_probability": 1.0,
                "text": "one two three",
                "words": [
                    {"text": "one", "start": 0, "end": 9.5, "type": "word"},
                    {"text": " ", "start": 9.5, "end": 9.5, "type": "spacing"},
                    {"text": "two", "start": 9.5, "end": 10, "type": "word"},
                    {"text": " ", "start": 10, "end": 10, "type": "spacing"},
                    {"text": "three", "start": 10, "end": 11, "type": "word"},
                ],
            })

        with httpx.Client(transport=httpx.MockTransport(respond)) as http:
            client = ElevenLabs(api_key="test-key", httpx_client=http)
            with tempfile.TemporaryDirectory() as directory:
                audio = Path(directory) / "audio.wav"
                audio.write_bytes(b"same WAV bytes")
                with patch.dict(os.environ, {"SKIM_STT_PROVIDER": "elevenlabs", "ELEVENLABS_API_KEY": "test-key"}), patch("elevenlabs.client.ElevenLabs", return_value=client):
                    self.assertEqual(stt.transcribe(str(audio)), [stt.Segment(0, 10, "one two"), stt.Segment(10, 11, "three")])
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].url.path, "/v1/speech-to-text")
        self.assertEqual(requests[0].headers["xi-api-key"], "test-key")
        self.assertIn(b"same WAV bytes", requests[0].content)
        self.assertIn(b"scribe_v2", requests[0].content)
        self.assertNotIn(b'name="language_code"', requests[0].content)


class EvaluationTests(unittest.TestCase):
    def test_transcript_caches_are_isolated_and_visuals_identical(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            cache = Path(directory) / ".cache"
            video = Path(directory) / "ted.mp4"
            video.write_bytes(b"video")
            stack.enter_context(patch.object(ev, "CACHE_DIR", cache))
            stack.enter_context(patch.object(ev, "extract_audio"))
            frames = stack.enter_context(patch.object(ev, "extract_frames", return_value=[]))
            describe = stack.enter_context(patch.object(ev, "describe_frames", return_value=[FrameDescription(0, "same visuals")]))
            transcribe = stack.enter_context(patch.object(ev, "transcribe", side_effect=lambda _: [stt.Segment(0, 1, stt.get_stt_provider())]))
            stack.enter_context(patch.object(ev, "build_index", side_effect=lambda segments, visuals: (segments, visuals)))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            with patch.dict(os.environ, {"SKIM_STT_PROVIDER": "whisper"}):
                whisper = ev.ingest_video(str(video))
                whisper_path = ev._cache_path(str(video))
            with patch.dict(os.environ, {"SKIM_STT_PROVIDER": "elevenlabs"}):
                elevenlabs = ev.ingest_video(str(video))
                elevenlabs_path = ev._cache_path(str(video))
                self.assertEqual(ev.ingest_video(str(video)), elevenlabs)
            self.assertNotEqual(whisper_path, elevenlabs_path)
            self.assertEqual(whisper[0][0].text, "whisper")
            self.assertEqual(elevenlabs[0][0].text, "elevenlabs")
            self.assertEqual(whisper[1], elevenlabs[1])
            self.assertEqual(transcribe.call_count, 2)
            frames.assert_called_once()
            describe.assert_called_once()
            with patch.dict(os.environ, {"SKIM_STT_PROVIDER": "elevenlabs"}):
                video.write_bytes(b"changed video")
                self.assertNotEqual(ev._cache_path(str(video)), elevenlabs_path)

    def test_legacy_cache_supplies_only_visuals(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(ev, "CACHE_DIR", Path(directory)), patch.dict(os.environ, {"SKIM_STT_PROVIDER": "elevenlabs"}):
            video = Path(directory) / "ted.mp4"
            video.write_bytes(b"video")
            path = ev._cache_path(str(video))
            legacy = {"cache_version": ev.CACHE_VERSION, "segments": [{"start": 0, "end": 1, "text": "legacy whisper"}], "frame_descriptions": [{"timestamp": 0, "description": "legacy visual"}]}
            (Path(directory) / path.name).write_text(json.dumps(legacy), encoding="utf-8")
            with patch.object(ev, "extract_audio"), patch.object(ev, "transcribe", return_value=[stt.Segment(0, 1, "new elevenlabs")]) as transcribe, patch.object(ev, "extract_frames") as frames, patch.object(ev, "build_index", side_effect=lambda segments, visuals: (segments, visuals)):
                segments, visuals = ev.ingest_video(str(video))
            self.assertEqual(segments[0].text, "new elevenlabs")
            self.assertEqual(visuals[0].description, "legacy visual")
            transcribe.assert_called_once()
            frames.assert_not_called()

    def test_full_and_ted_runs_have_separate_provider_results(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            original_dataset = json.loads(ev.DATASET_PATH.read_text(encoding="utf-8"))
            self.assertEqual(len(original_dataset["videos"]), 7)
            self.assertEqual(sum(len(v["questions"]) for v in original_dataset["videos"]), 19)
            dataset = Path(directory) / "dataset.json"
            dataset.write_text(json.dumps(original_dataset), encoding="utf-8")
            stack.enter_context(patch.object(ev, "DATASET_PATH", dataset))
            stack.enter_context(patch.object(ev, "__file__", str(Path(directory) / "run_evals.py")))
            ingest = stack.enter_context(patch.object(ev, "ingest_video", return_value=SimpleNamespace(items=[])))
            stack.enter_context(patch.object(ev, "retrieve", return_value=[]))
            stack.enter_context(patch.object(ev, "answer_question", return_value="answer"))
            stack.enter_context(patch.object(ev, "answer_naive", return_value="answer"))
            stack.enter_context(patch.object(ev, "judge", return_value={"verdict": "correct", "reasoning": "test"}))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            for provider in ("whisper", "elevenlabs"):
                with patch.dict(os.environ, {"SKIM_STT_PROVIDER": provider}):
                    self.assertEqual(len(ev.run()), 38)
                    full = (Path(directory) / f"results_{provider}.json").read_bytes()
                    ingest.reset_mock()
                    rows = ev.run(["ted"])
                    self.assertEqual({r["video"] for r in rows}, {"ted"})
                    self.assertEqual({r["stt_provider"] for r in rows}, {provider})
                    ingest.assert_called_once()
                    self.assertTrue((Path(directory) / f"results_{provider}_ted.json").exists())
                    self.assertEqual((Path(directory) / f"results_{provider}.json").read_bytes(), full)
                    with self.assertRaises(ValueError):
                        ev.run(["typo"])


if __name__ == "__main__":
    unittest.main()
