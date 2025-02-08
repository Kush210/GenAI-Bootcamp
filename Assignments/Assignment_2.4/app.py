from openai import OpenAI
import os
import streamlit as st
from langchain.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pytube import YouTube
import tempfile

# from langchain.document_loaders import WebBaseLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from transformers import pipeline

# from pydub import AudioSegment
import yt_dlp
from dotenv import load_dotenv
import requests
from PIL import Image
from io import BytesIO

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
# Set options for yt-dlp
ydl_opts = {
    "format": "bestaudio/best",  # Select the best audio quality
    "outtmpl": "./audio",  # Output file name with .wav extension
    "postprocessors": [
        {  # Extract audio using ffmpeg and save as WAV
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",  # Bitrate doesn't apply to WAV but is kept for consistency
        }
    ],
    "ffmpeg_location": "/usr/bin/ffmpeg",
}

# Ollama model for Llama3
llm_llama3 = Ollama(model="llama3")
output_parser = StrOutputParser()

# streamlit app showing title and description

st.title("Video to blog post")

st.write("This app converts a YouTube video into a blog post.")

YouTube_video = st.text_input("Enter the YouTube video URL:")

if YouTube_video:
    with st.spinner("Downloading video..."):
        # Create a YouTube object with the URL
        yt = YouTube(YouTube_video)

        # Get the publication date
        Date = yt.publish_date

        # Get the highest quality audio stream
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([YouTube_video])
        # Function to download YouTube video and extract audio

    with st.spinner("Transcribing audio..."):
        whisper = pipeline(
            "automatic-speech-recognition", model="openai/whisper-medium", device=0
        )

        transcript = whisper("audio.wav", return_timestamps=True)["text"]
        print(transcript[:200])

    with st.spinner("Generating article..."):
        # create a spinned article from this video transcription. Create a title, subtitle and a relevant image for the spinned article.
        system_prompt = """You are an assitant which creates a blog article from provided transcription of a video. The blog should contain appropriate title and subtitle. 
                        Use markdown fromat for formatting the article.\n\n
                        """

        prompt = ChatPromptTemplate(
            messages=[
                ("system", system_prompt),
                ("user", "{transcript}"),
            ]
        )

        chain = prompt | llm_llama3 | StrOutputParser()

        # Invoke the chain to process the input
        article = chain.invoke({"transcript": transcript})

        # Output the result (the spinned article with title, subtitle, and image prompt)
        print(article)
        st.markdown(article)

    with st.spinner("Generating image..."):
        img_system_prompt = """You are an assitant which creates a image prompt  from provided blog article.Only give the image prompt\n\n """

        prompt = ChatPromptTemplate(
            messages=[
                ("system", img_system_prompt),
                ("user", "{article}"),
            ]
        )

        chain = prompt | llm_llama3 | StrOutputParser()

        # Invoke the chain to process the input
        img_prompt = chain.invoke({"article": article})

        img = client.images.generate(
            model="dall-e-3", prompt=img_prompt, size="1024x1024", n=1
        )
        img_url = img.data[0].url
        image = requests.get(img_url)
        image = Image.open(BytesIO(image.content))

        st.image(image, caption="Generated Image", use_container_width=True)

    with st.spinner("Generating dictation..."):
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=transcript,
        )

        st.audio(response.content, format="audio/mpeg")

    st.success("Video downloaded and processed successfully!")
