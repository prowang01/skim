"""Palier 3: upload an mp4, transcribe its speech, extract + describe key
visual frames, index everything, and chat with retrieval-driven answers."""

import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from ingest.extract import extract_audio
from ingest.transcribe import transcribe
from ingest.frames import extract_frames
from ingest.describe import describe_frames
from index.build_index import build_index
from index.retrieve import retrieve
from qa.answer import answer_question

load_dotenv()

st.set_page_config(page_title="Skim — chat with a video's speech and visuals")
st.title("Skim")
st.caption(
    "Palier 3 (retrieval): upload a talking-style video (tutorial, talk, "
    "interview) and ask questions about what was said and shown. Each question "
    "retrieves the relevant moments instead of using the whole video, and "
    "answers cite [mm:ss] timestamps."
)

uploaded_file = st.file_uploader("Upload an .mp4 file", type=["mp4"])

if uploaded_file is not None:
    if st.session_state.get("processed_filename") != uploaded_file.name:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            video_path = tmp_path / uploaded_file.name
            audio_path = tmp_path / "audio.wav"
            frames_dir = tmp_path / "frames"
            video_path.write_bytes(uploaded_file.getvalue())

            with st.spinner("Extracting audio with ffmpeg..."):
                extract_audio(str(video_path), str(audio_path))

            with st.spinner("Transcribing with faster-whisper (first run downloads the model)..."):
                segments = transcribe(str(audio_path))

            with st.spinner("Detecting scene changes and extracting frames..."):
                frames = extract_frames(str(video_path), str(frames_dir))

            with st.spinner(f"Describing {len(frames)} frame(s) with GPT-4o..."):
                frame_descriptions = describe_frames(frames)

            with st.spinner("Building the semantic index..."):
                index = build_index(segments, frame_descriptions)

            # Snapshot thumbnails to bytes now -- the tempdir (and the frame
            # jpegs in it) goes away as soon as this block exits.
            frame_previews = [
                (fd.timestamp, Path(f.path).read_bytes(), fd.description)
                for f, fd in zip(frames, frame_descriptions)
            ]

        st.session_state["segments"] = segments
        st.session_state["frame_descriptions"] = frame_descriptions
        st.session_state["frame_previews"] = frame_previews
        st.session_state["index"] = index
        st.session_state["processed_filename"] = uploaded_file.name
        st.session_state["chat_history"] = []

if "segments" in st.session_state:
    segments = st.session_state["segments"]
    frame_descriptions = st.session_state["frame_descriptions"]
    frame_previews = st.session_state["frame_previews"]

    with st.expander(f"Transcript ({len(segments)} segments)"):
        for s in segments:
            minutes, secs = divmod(int(s.start), 60)
            st.text(f"[{minutes:02d}:{secs:02d}] {s.text}")

    with st.expander(f"Visual frames ({len(frame_previews)} sampled)"):
        for timestamp, image_bytes, description in frame_previews:
            minutes, secs = divmod(int(timestamp), 60)
            st.image(image_bytes, caption=f"[{minutes:02d}:{secs:02d}] {description}", width=320)

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
            with st.spinner("Retrieving relevant moments..."):
                retrieved = retrieve(st.session_state["index"], question)
            with st.spinner("Thinking..."):
                answer = answer_question(
                    retrieved,
                    question,
                    history=st.session_state["chat_history"][:-1],
                )
            st.markdown(answer)
            with st.expander(f"Retrieved for this question ({len(retrieved)} items)"):
                for item in retrieved:
                    minutes, secs = divmod(int(item.timestamp), 60)
                    st.text(f"[{minutes:02d}:{secs:02d}] ({item.kind}) {item.text}")
        st.session_state["chat_history"].append({"role": "assistant", "content": answer})
else:
    st.info("Upload an mp4 to get started.")
