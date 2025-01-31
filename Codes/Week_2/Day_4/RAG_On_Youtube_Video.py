from dotenv import load_dotenv
import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
import yt_dlp

# Function to download audio and video using youtube-dl
def download_youtube(url):
    """
    Download both video and audio from a YouTube URL and save as MP3 and MP4 files.

    Args:
    url (str): The URL of the YouTube video.

    Returns:
    tuple: Filenames of the downloaded MP3 and MP4 files.
    """
    # Options for downloading audio
    ydl_opts_audio = {
        'format': 'bestaudio/best',
        'outtmpl': 'audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    
    # Options for downloading video
    ydl_opts_video = {
        'format': 'bestvideo+bestaudio',
        'outtmpl': 'video.%(ext)s',
    }
    
    # Download audio
    with yt_dlp.YoutubeDL(ydl_opts_audio) as ydl:
        ydl.download([url])
    
    # Download video
    with yt_dlp.YoutubeDL(ydl_opts_video) as ydl:
        ydl.download([url])
    
    return 'audio.mp3', 'video.mp4'

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment variable
openai_key = os.getenv('OPENAI_KEY')
# Add a heading to the Streamlit page
st.title("Chat with an Youtube Video")

# Streamlit input for the YouTube video URL
YOUTUBE_VIDEO = st.text_input("Enter YouTube video URL")

# Check if the video URL is provided
if YOUTUBE_VIDEO:
   # Download audio and video
    audio_stream, video_stream = download_youtube(YOUTUBE_VIDEO)
    
    # Save the video and audio files
    st.write(f"Audio file saved as: {audio_stream}")
    st.write(f"Video file saved as: {video_stream}")
    
    # Display the audio and video files
    st.audio(audio_stream)
    st.video(video_stream)

    # OpenAI API key (ensure this is set in your environment)
    OPENAI_KEY = os.getenv('OPENAI_API_KEY')

    model = ChatOpenAI(openai_api_key=OPENAI_KEY, model="gpt-4o-mini")
    parser = StrOutputParser()

    template = """
    Answer the question based on the context below. If you can't
    answer the question, reply "I don't know".

    Context: {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    client = OpenAI(api_key=OPENAI_KEY)
    st.write('Download completed! Transcribing through OpenAI')

    with open(audio_stream, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
    st.write('Transcript:')
    st.write(transcript)

    # Save transcript to transcription.txt
    with open("transcription.txt", "w") as file:
        file.write(transcript)

    with open("transcription.txt") as file:
        transcription = file.read()

    loader = TextLoader("transcription.txt")
    text_documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    documents = text_splitter.split_documents(text_documents)

    embeddings = OpenAIEmbeddings(api_key=OPENAI_KEY)

    st.write("Indexing chunks...")
    vectorstore1 = DocArrayInMemorySearch.from_documents(documents, embeddings)
    retriever1 = vectorstore1.as_retriever()

    setup = RunnableParallel(context=retriever1, question=RunnablePassthrough())

    st.write("Indexing complete. You can ask your questions now.")
    user_question = st.text_input("Ask your question")

    if user_question:
        chain = (
            {"context": vectorstore1.as_retriever(), "question": RunnablePassthrough()}
            | prompt
            | model
            | parser
        )
        answer = chain.invoke(user_question)
        st.write('Answer:')
        st.write(answer)

        # Loop for additional questions
        count =0;
        while True:
            more_questions = st.text_input("Ask another question", key='more_questions_' + str(count))
            if more_questions:
                answer = chain.invoke(more_questions)
                st.write('Answer:')
                st.write(answer)
                count+=1
            else:
                break


