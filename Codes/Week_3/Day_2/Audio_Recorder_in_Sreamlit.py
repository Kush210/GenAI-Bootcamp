<<<<<<< HEAD
import streamlit as st
from audiorecorder import audiorecorder


st.title("Audio Recorder")
audio = audiorecorder("Click to record", "Click to stop recording")

if len(audio) > 0:
    # To play audio in frontend:
=======
import streamlit as st
from audiorecorder import audiorecorder


st.title("Audio Recorder")
audio = audiorecorder("Click to record", "Click to stop recording")

if len(audio) > 0:
    # To play audio in frontend:
>>>>>>> vettura/main
    st.audio(audio.export().read())