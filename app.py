"""Palier 1: upload an mp4, transcribe its speech, chat about what was said."""

import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from ingest.extract import extract_audio
from ingest.transcribe import transcribe
from qa.answer import answer_question

load_dotenv()

st.set_page_config(page_title="Skim — chat with a video's speech")
st.title("Skim")
st.caption(
    "Palier 1 (audio MVP): upload a talking-style video (tutorial, talk, interview) "
    "and ask questions about what was said. Answers cite [mm:ss] timestamps."
)

uploaded_file = st.file_uploader("Upload an .mp4 file", type=["mp4"])

if uploaded_file is not None:
    if st.session_state.get("processed_filename") != uploaded_file.name:
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / uploaded_file.name
            audio_path = Path(tmp_dir) / "audio.wav"
            video_path.write_bytes(uploaded_file.getvalue())

            with st.spinner("Extracting audio with ffmpeg..."):
                extract_audio(str(video_path), str(audio_path))

            with st.spinner("Transcribing with faster-whisper (first run downloads the model)..."):
                segments = transcribe(str(audio_path))

        st.session_state["segments"] = segments
        st.session_state["processed_filename"] = uploaded_file.name
        st.session_state["chat_history"] = []

if "segments" in st.session_state:
    segments = st.session_state["segments"]

    with st.expander(f"Transcript ({len(segments)} segments)"):
        for s in segments:
            minutes, secs = divmod(int(s.start), 60)
            st.text(f"[{minutes:02d}:{secs:02d}] {s.text}")

    st.divider()
    st.subheader("Ask about the video")

    for turn in st.session_state["chat_history"]:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    question = st.chat_input("What do you want to know about this video?")
    if question:
        st.session_state["chat_history"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = answer_question(
                    segments, question, history=st.session_state["chat_history"][:-1]
                )
            st.markdown(answer)
        st.session_state["chat_history"].append({"role": "assistant", "content": answer})
else:
    st.info("Upload an mp4 to get started.")
