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


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("Ollama Chat App")

# {
#     "role" : "user/assistant"
#     "content" :"response/query"
# }

# [
#     p1,p2,p3
# ]

def generate_response(user_input):

    st.session_state.chat_history.append(
        {
            "role" :"user",
            "content" : user_input
        }
    )

    response = query(user_input)
    
    st.session_state.chat_history.append(
        {
            "role" :"assistant",
            "content" : response['content']
        }
    )

    for message in st.session_state.chat_history:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

if user_input := st.chat_input("Your Message"):
    generate_response(user_input)