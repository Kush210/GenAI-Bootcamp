import streamlit as st 
import requests as re 

def query(user_input):
    url = 'http://localhost:11434/api/chat'
    payload = {
        "model" : "gemma:2b",
        "messages" : [{
            "role" : "user",
            "content" : user_input
        }],
        "stream" : False
    }

    response = re.post(url = url, json = payload)

    return response.json()['message']


st.title("Ollama Chat App")

if user_input := st.chat_input("Your Message"):
    with st.chat_message("user"):
        st.markdown(user_input)

    response = query(user_input)

    with st.chat_message("assistant"):
        st.markdown(response['content'])

