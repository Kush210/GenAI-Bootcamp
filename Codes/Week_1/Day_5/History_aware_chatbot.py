import streamlit as st
import requests as re

# Function to send query to the API
@st.cache_data
def query(user_input, context):
    url = 'http://localhost:11434/api/chat'

    system = {
        "role" : "system",
        "content" : "Your are Vettura, an obedient and helpful AI assistant. Help the user with his questions and be nice to the user."
    }
    
    messages  = []

    messages.append(system)
    for msg in context:
        messages.append(msg)
    
    user = {
        "role" : "user",
        "content" : user_input
    }
    
    messages.append(user)

    st.write(messages)

    payload = {
        "model": "gemma:2b",
        "messages": messages,
        "stream": False,
        "options" : {
            "temperature" : 0.5
        }
    }

    response = re.post(url=url, json=payload)
    return response.json()  # Return the full response to include detailed explanations


# Initialize the chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("Vettura Chat")

# Roles explained:
# "system": Represents system-level instructions or configurations for the bot.
# "user": Represents the user's messages and queries in the conversation.
# "assistant": Represents the bot's responses to the user's queries.

# Function to generate a response and maintain conversation history
def generate_response(user_input):
    # Include the last 10 messages as context for the conversation
    context = st.session_state.chat_history[-5:]  # Take the last 5 messages only
    # context
    # Query the API with the user's input and context
    response = query(user_input, context)

    # Append the user's input to the chat history
    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # Append the bot's response to the chat history
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": response['message']['content']  # Assuming API response contains a 'message' key
        }
    )

    # Display the conversation on the UI
    for message in st.session_state.chat_history:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    
    
    # st.session_state.chat_history

# Input box for user message
if user_input := st.chat_input("Your Message"):
    generate_response(user_input)
