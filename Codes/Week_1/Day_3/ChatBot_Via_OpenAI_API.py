# Steps to Create and Use .env for API Keys
#   1. pip install python-dotenv
#   2. Create the .env File: In the root directory of your project (where your Python script is),\
#      create a new file called .env. This file will hold your environment variables in the format \n 
#      of KEY=VALUE. Example: OPENAI_API_KEY="your_openai_api_key_here"


from dotenv import load_dotenv 
# loading variables from .env file
load_dotenv() 

import streamlit as st
from openai import Client

client=Client()

def query(user_input): 
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input}
        ],
        temperature=0.7,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response.choices[0].message.content

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("OpenAI Chat App")

def generate_response(user_input):
    st.session_state.chat_history.append(
        {"role": "user", "content": user_input}
    )

    response = query(user_input)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": response}
    )

    for message in st.session_state.chat_history:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

if user_input := st.chat_input("Your Message"):
    with st.spinner("Generating response..."):
        generate_response(user_input)